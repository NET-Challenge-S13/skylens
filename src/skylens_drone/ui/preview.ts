// What the "camera" is currently pointed at.
//
// In demo mode the camera is a folder of recorded clips (COMPONENTS.md §5.1), so
// the honest preview is the clip that the last slice resolved to. The files are
// 4K/60 and 26–38 MB each, which will stutter a demo machine that is also
// rendering the control tower, so playback is OPT-IN: by default this shows the
// filename and metadata only and fetches nothing.

export class CameraPreview {
  private root: HTMLElement;
  private video: HTMLVideoElement;
  private caption: HTMLElement;
  private toggle: HTMLInputElement;
  private uri: string | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
    this.root.className = 'preview';

    const head = document.createElement('header');
    head.className = 'preview__head';
    head.textContent = '카메라 (데모 대체 영상)';

    const label = document.createElement('label');
    label.className = 'preview__toggle';
    this.toggle = document.createElement('input');
    this.toggle.type = 'checkbox';
    label.append(this.toggle, document.createTextNode(' 재생 (4K 파일, 무거움)'));
    head.append(label);

    this.video = document.createElement('video');
    this.video.className = 'preview__v';
    this.video.muted = true;
    this.video.loop = true;
    this.video.playsInline = true;
    this.video.preload = 'none';
    this.video.hidden = true;

    this.caption = document.createElement('div');
    this.caption.className = 'preview__cap';
    this.caption.textContent = '아직 촬영 구간 없음';

    this.root.append(head, this.video, this.caption);
    this.toggle.addEventListener('change', () => this.apply());
  }

  /** Called each time a slice resolves to a clip. */
  show(uri: string, note: string): void {
    this.caption.textContent = note;
    if (this.uri === uri) return;
    this.uri = uri;
    this.apply();
  }

  private apply(): void {
    const on = this.toggle.checked && this.uri !== null;
    this.video.hidden = !on;
    if (!on) {
      this.video.pause();
      this.video.removeAttribute('src');
      this.video.load();
      return;
    }
    if (this.video.getAttribute('src') !== this.uri) {
      this.video.src = this.uri as string;
    }
    void this.video.play().catch(() => {
      this.caption.textContent += ' — 브라우저가 자동 재생을 막았습니다';
    });
  }
}
