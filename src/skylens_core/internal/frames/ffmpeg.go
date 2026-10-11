package frames

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
	"github.com/NET-Challenge-S13/skylens/src/skylens_core/internal/backoff"
)

// FFmpeg reads one live stream with an ffmpeg subprocess and emits JPEG
// frames at FPS. ffmpeg is restarted with backoff when it exits (the stream
// may not be published yet, or the bridge pull dropped).
type FFmpeg struct {
	Bin     string  // default "ffmpeg"
	URL     string  // input, e.g. srt://127.0.0.1:10890?streamid=read:drone1
	FPS     float64 // frames per second to extract; default 2
	Quality int     // mjpeg -q:v, 2 (best) .. 31; default 5
	// MaxFrameBytes guards against a runaway stream; default 16 MiB.
	MaxFrameBytes int
	Backoff       backoff.Backoff
}

// Args is the ffmpeg command line (without the binary).
func (f *FFmpeg) Args() []string {
	fps := f.FPS
	if fps <= 0 {
		fps = 2
	}
	q := f.Quality
	if q <= 0 {
		q = 5
	}
	return []string{
		"-hide_banner", "-loglevel", "error", "-nostdin",
		"-fflags", "nobuffer", "-flags", "low_delay",
		"-i", f.URL,
		"-an",
		"-vf", "fps=" + strconv.FormatFloat(fps, 'f', -1, 64),
		"-c:v", "mjpeg", "-q:v", strconv.Itoa(q),
		"-f", "image2pipe", "pipe:1",
	}
}

// Run keeps ffmpeg running until ctx is done.
func (f *FFmpeg) Run(ctx context.Context, emit func(Frame)) error {
	bin := f.Bin
	if bin == "" {
		bin = "ffmpeg"
	}
	if _, err := exec.LookPath(bin); err != nil {
		return fmt.Errorf("ffmpeg not found (%s): %w", bin, err)
	}
	for ctx.Err() == nil {
		started := time.Now()
		n, err := f.once(ctx, bin, emit)
		if ctx.Err() != nil {
			return nil
		}
		if n > 0 || time.Since(started) > 30*time.Second {
			f.Backoff.Reset()
		}
		log.Printf("[core] ffmpeg %s exited after %d frame(s): %v", f.URL, n, err)
		if !f.Backoff.Wait(ctx) {
			return nil
		}
	}
	return nil
}

func (f *FFmpeg) once(ctx context.Context, bin string, emit func(Frame)) (int, error) {
	cmd := exec.CommandContext(ctx, bin, f.Args()...)
	var stderr bytes.Buffer
	cmd.Stderr = &limitedWriter{w: &stderr, n: 4096}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	n := 0
	splitErr := SplitJPEG(out, f.MaxFrameBytes, func(jpeg []byte) {
		n++
		emit(Frame{
			Format:     modelv1.ImageFormat_IMAGE_FORMAT_JPEG,
			Data:       jpeg,
			CapturedAt: time.Now(),
		})
	})
	if splitErr != nil {
		// Unblock ffmpeg if we stopped reading early.
		_, _ = io.Copy(io.Discard, out)
	}
	waitErr := cmd.Wait()
	if msg := strings.TrimSpace(stderr.String()); msg != "" {
		return n, fmt.Errorf("%v: %s", waitErr, msg)
	}
	if splitErr != nil {
		return n, splitErr
	}
	if waitErr == nil {
		waitErr = errors.New("stream ended")
	}
	return n, waitErr
}

type limitedWriter struct {
	w io.Writer
	n int
}

func (l *limitedWriter) Write(p []byte) (int, error) {
	if l.n <= 0 {
		return len(p), nil
	}
	q := p
	if len(q) > l.n {
		q = q[:l.n]
	}
	l.n -= len(q)
	_, _ = l.w.Write(q)
	return len(p), nil
}

// ErrFrameTooLarge is returned when a JPEG exceeds the size cap.
var ErrFrameTooLarge = errors.New("frames: jpeg frame exceeds the size cap")

// SplitJPEG reads concatenated JPEG images (ffmpeg image2pipe output) and calls
// fn with each one. Images are delimited by SOI (FF D8) and EOI (FF D9). In the
// entropy-coded data a literal FF is always stuffed as FF 00, so FF D9 only
// occurs as a marker. Bytes before an SOI are skipped. It returns nil at a
// clean EOF.
func SplitJPEG(r io.Reader, maxBytes int, fn func([]byte)) error {
	if maxBytes <= 0 {
		maxBytes = 16 << 20
	}
	br := bufio.NewReaderSize(r, 256<<10)
	var cur []byte
	in := false
	var prev byte
	havePrev := false
	for {
		b, err := br.ReadByte()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if !in {
			if havePrev && prev == 0xFF && b == 0xD8 {
				in = true
				cur = append(cur[:0], 0xFF, 0xD8)
				havePrev = false
				continue
			}
			prev, havePrev = b, true
			continue
		}
		cur = append(cur, b)
		if len(cur) > maxBytes {
			return ErrFrameTooLarge
		}
		n := len(cur)
		if cur[n-2] == 0xFF && b == 0xD9 {
			img := make([]byte, n)
			copy(img, cur)
			fn(img)
			in = false
			havePrev = false
			cur = cur[:0]
		}
	}
}
