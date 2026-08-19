// Rolling event log. The same lines the headless runner prints to stdout, so a
// transcript captured from the browser is comparable with one from the terminal.

const MAX_LINES = 200;

export class LogView {
  private root: HTMLElement;
  private list: HTMLElement;
  private started = Date.now();

  constructor(root: HTMLElement) {
    this.root = root;
    this.root.className = 'logview';
    const head = document.createElement('header');
    head.className = 'logview__head';
    head.textContent = '이벤트 로그';
    this.list = document.createElement('ol');
    this.list.className = 'logview__list';
    this.root.append(head, this.list);
  }

  push(line: string): void {
    const item = document.createElement('li');
    const stamp = document.createElement('span');
    stamp.className = 'logview__t';
    stamp.textContent = `+${((Date.now() - this.started) / 1000).toFixed(1)}s`;
    const body = document.createElement('span');
    body.className = 'logview__m';
    body.textContent = line;
    item.append(stamp, body);
    this.list.append(item);
    while (this.list.childElementCount > MAX_LINES) this.list.firstElementChild?.remove();
    this.list.scrollTop = this.list.scrollHeight;
  }
}
