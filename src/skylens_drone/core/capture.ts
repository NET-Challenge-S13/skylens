// Camera -> H.265 slices.
//
// The drone cuts its stream into fixed slices and ships each one with the poses
// covering it (protocol.ts §2, VideoSegment). Two implementations of the same
// interface:
//
//   LiveCapture — a real WebCodecs H.265 encode of the camera track. This is
//                 the path a flying drone takes. It needs `VideoEncoder` with an
//                 hvc1/hev1 config and `MediaStreamTrackProcessor`; where those
//                 are missing (Node, Firefox, most desktops without a hardware
//                 HEVC encoder) it refuses to start rather than pretending.
//
//   DemoCapture — COMPONENTS.md §5.1 substitution: there is no camera, so each
//                 slice resolves to one of the pre-recorded clips under
//                 res/static/video/h265. Nothing is encoded AT CUT TIME — the
//                 encode happened once, ahead of the demo, via
//                 tools/transcodeDemoFootage.ts. The files really are HEVC, so
//                 the `codec` on the wire is not a stand-in for anything.

import { pickClip, wireCodecOf, type FlightDirection } from './demoAssets.ts';

export interface SliceRequest {
  /** Slice number within the flight, 0-based. */
  index: number;
  /** Where the slice sat along the route, 0..1 (midpoint). */
  fraction: number;
  direction: FlightDirection;
  startedAt: number;
  durationMs: number;
}

export interface SliceResult {
  uri: string;
  bytes: number;
  /**
   * Codec of the bytes THIS slice actually consists of, established per slice
   * rather than declared once by the source. VideoSegment.codec is copied from
   * here, so a segment can never claim a codec nothing verified.
   */
  codec: 'h265';
  /** Human-readable note for the operator panel / logs. */
  note: string;
}

export interface CaptureSource {
  readonly kind: 'demo' | 'live';
  start(): Promise<void>;
  stop(): void;
  cutSlice(req: SliceRequest): Promise<SliceResult>;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

export class DemoCapture implements CaptureSource {
  readonly kind = 'demo' as const;

  async start(): Promise<void> {
    // Nothing to open: the "camera" is a folder of files.
  }

  stop(): void {}

  async cutSlice(req: SliceRequest): Promise<SliceResult> {
    const clip = pickClip(req.fraction, req.direction);
    // Throws rather than mislabels if the footage was never transcoded.
    const codec = wireCodecOf(clip);
    return {
      uri: clip.uri,
      bytes: clip.bytes,
      codec,
      note:
        `demo footage ${clip.file} (${clip.width}x${clip.height} ${codec}, ` +
        `${(clip.bytes / 1e6).toFixed(1)} MB, pre-encoded)`,
    };
  }
}

// ---------------------------------------------------------------------------
// Live (WebCodecs)
// ---------------------------------------------------------------------------

// Structural stand-ins rather than ambient declarations: WebCodecs is only
// partly in lib.dom and MediaStreamTrackProcessor is not in it at all, so
// declaring globals here would clash on some TS versions. Everything is fetched
// off globalThis and shaped locally.

interface EncodedChunkLike {
  byteLength: number;
  type: 'key' | 'delta';
  copyTo(dest: Uint8Array): void;
}

interface VideoEncoderLike {
  encode(frame: unknown, opts?: { keyFrame?: boolean }): void;
  configure(config: Record<string, unknown>): void;
  flush(): Promise<void>;
  close(): void;
}

interface VideoEncoderCtor {
  new (init: {
    output: (chunk: EncodedChunkLike) => void;
    error: (err: unknown) => void;
  }): VideoEncoderLike;
  isConfigSupported(config: Record<string, unknown>): Promise<{ supported?: boolean }>;
}

interface TrackProcessorCtor {
  new (init: { track: unknown }): { readable: ReadableStream<{ close(): void }> };
}

export interface LiveCaptureOptions {
  /** Camera track to encode. */
  track: unknown;
  width: number;
  height: number;
  framerate: number;
  bitrate: number;
  /** Main profile, level 4.1 by default — what a 5G drone uplink would use. */
  codecString?: string;
  /** Turn a finished slice into something the core can fetch. Injected so this
   *  file stays free of upload policy: the Tauri shell writes to disk, a plain
   *  browser makes a blob: URL. */
  publish: (bytes: Uint8Array, req: SliceRequest) => Promise<string>;
}

export class LiveCapture implements CaptureSource {
  readonly kind = 'live' as const;

  private opts: LiveCaptureOptions;
  private encoder: VideoEncoderLike | null = null;
  private pending: EncodedChunkLike[] = [];
  private reading = false;

  constructor(opts: LiveCaptureOptions) {
    this.opts = opts;
  }

  static available(): boolean {
    const g = globalThis as Record<string, unknown>;
    return typeof g.VideoEncoder === 'function' && typeof g.MediaStreamTrackProcessor === 'function';
  }

  async start(): Promise<void> {
    const g = globalThis as unknown as {
      VideoEncoder?: VideoEncoderCtor;
      MediaStreamTrackProcessor?: TrackProcessorCtor;
    };
    if (!g.VideoEncoder || !g.MediaStreamTrackProcessor) {
      throw new Error('WebCodecs H.265 encoding is unavailable in this runtime');
    }
    const config = {
      codec: this.opts.codecString ?? 'hev1.1.6.L123.B0',
      width: this.opts.width,
      height: this.opts.height,
      framerate: this.opts.framerate,
      bitrate: this.opts.bitrate,
      latencyMode: 'realtime',
    };
    const probe = await g.VideoEncoder.isConfigSupported(config);
    if (probe.supported === false) {
      throw new Error(`no H.265 encoder for ${config.codec} at ${config.width}x${config.height}`);
    }

    const encoder = new g.VideoEncoder({
      output: (chunk) => this.pending.push(chunk),
      error: (err) => {
        this.pending = [];
        console.error('[drone] encoder error', err);
      },
    });
    encoder.configure(config);
    this.encoder = encoder;

    const processor = new g.MediaStreamTrackProcessor({ track: this.opts.track });
    this.reading = true;
    void this.pump(processor.readable, encoder);
  }

  private async pump(readable: ReadableStream<{ close(): void }>, encoder: VideoEncoderLike): Promise<void> {
    const reader = readable.getReader();
    let n = 0;
    while (this.reading) {
      const { value, done } = await reader.read();
      if (done || !value) break;
      // One keyframe at the head of every ~2 s so a slice can start decoding
      // on its own without the previous slice.
      encoder.encode(value, { keyFrame: n % Math.round(this.opts.framerate * 2) === 0 });
      value.close();
      n++;
    }
    reader.releaseLock();
  }

  stop(): void {
    this.reading = false;
    this.encoder?.close();
    this.encoder = null;
  }

  async cutSlice(req: SliceRequest): Promise<SliceResult> {
    const encoder = this.encoder;
    if (!encoder) throw new Error('capture not started');
    await encoder.flush();
    const chunks = this.pending;
    this.pending = [];
    const total = chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const bytes = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) {
      c.copyTo(bytes.subarray(at, at + c.byteLength));
      at += c.byteLength;
    }
    const uri = await this.opts.publish(bytes, req);
    // The encoder was configured with an hev1/hvc1 codec string and
    // isConfigSupported() agreed, so these bytes are H.265 by construction.
    return {
      uri,
      bytes: total,
      codec: 'h265',
      note: `live H.265 ${chunks.length} chunks / ${(total / 1e6).toFixed(2)} MB`,
    };
  }
}
