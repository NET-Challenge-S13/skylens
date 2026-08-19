// The one way a floating UI surface gets built.
//
// Both screens overlay the same kind of object on the 3D view: a translucent
// panel with a hairline border — pane titles, connection badges, readouts, the
// video frame, the minimap, modals. They used to be assembled ad hoc, each
// module writing its own element and re-declaring the same look in its own
// stylesheet, which is how a chip ended up with a different radius and font
// from the panel next to it. One builder, one stylesheet block (.sl-surface in
// style.css), so a change to the look lands everywhere at once.
//
// Position and size stay with the caller: WHERE a panel sits is its own
// business, WHAT it looks like is not.

export type SurfaceVariant = 'chip' | 'panel' | 'frame' | 'modal';

export interface SurfaceOptions {
  variant: SurfaceVariant;
  /** Extra class for the caller's own positioning/sizing rules. */
  className?: string;
  /** Element id, when something needs to find it. */
  id?: string;
  /** Mount target. Omit to leave the surface detached. */
  mount?: HTMLElement | null;
  /** Leading status dot (chips and panel headers use it). */
  dot?: boolean;
  /** Text for a chip, or the heading of a panel. */
  title?: string;
}

export interface Surface {
  root: HTMLElement;
  /** The status dot, when one was requested. */
  dot: HTMLSpanElement | null;
  /** The title element, when one was requested. */
  title: HTMLElement | null;
  dispose(): void;
}

export function createSurface(opts: SurfaceOptions): Surface {
  const root = document.createElement('div');
  root.className = `sl-surface sl-surface--${opts.variant}`;
  if (opts.className) root.classList.add(...opts.className.split(/\s+/).filter(Boolean));
  if (opts.id) root.id = opts.id;

  let dot: HTMLSpanElement | null = null;
  if (opts.dot) {
    dot = document.createElement('span');
    dot.className = 'sl-dot';
    root.appendChild(dot);
  }

  let title: HTMLElement | null = null;
  if (opts.title !== undefined) {
    // A chip IS its title, so the text goes straight in; a panel gets a heading
    // element its contents can sit under.
    title = document.createElement('span');
    title.className = opts.variant === 'chip' ? 'sl-surface__label' : 'sl-surface__title';
    title.textContent = opts.title;
    root.appendChild(title);
  }

  opts.mount?.appendChild(root);

  return {
    root,
    dot,
    title,
    dispose(): void {
      root.remove();
    },
  };
}

/**
 * The screen's own name, top-left over the view. Used to be a hand-written div
 * in each static html shell — the only piece of chrome not built like the rest,
 * and the one that looked different because of it.
 */
export function mountPaneLabel(pane: HTMLElement, text: string, tone?: string): Surface {
  return createSurface({
    variant: 'chip',
    className: `pane-label${tone ? ` pane-label--${tone}` : ''}`,
    mount: pane,
    dot: true,
    title: text,
  });
}
