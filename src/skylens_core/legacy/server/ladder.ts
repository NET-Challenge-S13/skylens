// The refinement ladder — 중간보고서 표 8 / 표 9.
//
// 표 8 measured quality against training steps on 540 frames; the curve is
// logarithmic, so 3% of the training budget (1,000 steps) already gives a
// readable spatial structure while the payload is 1.0 MB against 34.9 MB at
// 30,000. 표 9 turned that into the three levels the commander actually sees.
//
// Level numbers are 1-based positions in this list, NOT step counts: a viewer
// compares levels, and a higher one replaces a lower one for the same segment.

export interface LadderLevel {
  /** 1-based rung. */
  level: number;
  steps: number;
  /** 표 8/표 9 wording — what the commander can make out at this rung. */
  label: string;
}

/** 표 8 "상태" column, keyed by the step counts it measured. */
// Labels are shown inline in the board's segment ladder rows, so they have to
// stay short — the 표 8 full wording overflowed the panel box.
const MEASURED_LABEL = new Map<number, string>([
  [250, '형상 윤곽 식별'],
  [1000, '골격 배치'],
  [3500, '표면 형성'],
  [7000, '실용 품질'],
  [15000, '형상 개수 고정'],
  [30000, '최종 품질'],
]);

function labelFor(steps: number): string {
  const exact = MEASURED_LABEL.get(steps);
  if (exact !== undefined) return exact;
  // Not one of the measured points: fall back to the nearest measured rung at or
  // below it, so an operator-tuned ladder still says something honest.
  let best: string = '개략 형상';
  for (const [s, label] of MEASURED_LABEL) if (s <= steps) best = label;
  return best;
}

export function buildLadder(levelSteps: number[]): LadderLevel[] {
  return levelSteps.map((steps, i) => ({ level: i + 1, steps, label: labelFor(steps) }));
}

export function rung(ladder: LadderLevel[], level: number): LadderLevel | null {
  return ladder[level - 1] ?? null;
}

export function topLevel(ladder: LadderLevel[]): number {
  return ladder.length;
}
