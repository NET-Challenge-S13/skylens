// Detection markers (PROJECT.md §2-3 / §7). Positions are DERIVED from the actual
// scene cloud so they always sit on the reconstructed geometry regardless of which
// splat is loaded (and so they get revealed as drones sweep past). Kind/label/
// confidence stay authored. Both computers derive from the identical cloud.

import type { Detection, Vec3 } from '../../skylens_core/types';
import type { SceneData } from '../../skylens_core/data/sceneData.ts';
import { resolveMapSpec } from '../../skylens_core/data/terrainSource.ts';

interface DetectionTemplate {
  id: string;
  kind: 'person' | 'danger';
  label: string;
  confidence: number;
  /** Where to sample the cloud, as a fraction of the point count (0..1). */
  at: number;
}

/** Collapse scenario — the original splat/procedural scenes. */
const TEMPLATES: DetectionTemplate[] = [
  { id: 'p1', kind: 'person', label: '생존자 추정 · 구역 A', confidence: 0.86, at: 0.22 },
  { id: 'p2', kind: 'person', label: '생존자 추정 · 구역 B', confidence: 0.74, at: 0.63 },
  { id: 'd1', kind: 'danger', label: '붕괴 위험구역 · 중앙', confidence: 0.91, at: 0.45 },
];

/** Wildfire scenario — terrain scenes (?map=…, 울진·강릉 presets). Same ids/kinds
 *  so the reveal/confirm/net flow is untouched; only the story changes. */
const WILDFIRE_TEMPLATES: DetectionTemplate[] = [
  { id: 'p1', kind: 'person', label: '고립 위험 · 인가 밀집', confidence: 0.82, at: 0.22 },
  { id: 'p2', kind: 'person', label: '고립 위험 · 산간 마을', confidence: 0.71, at: 0.63 },
  { id: 'd1', kind: 'danger', label: '화선 추정 · 확산 방향 주의', confidence: 0.9, at: 0.45 },
];

/** Build detections by sampling deterministic points from the scene cloud. */
export function buildDetections(data: SceneData): Detection[] {
  const n = data.count;
  const templates = resolveMapSpec() ? WILDFIRE_TEMPLATES : TEMPLATES;
  return templates.map((tpl) => {
    const idx = Math.min(n - 1, Math.max(0, Math.floor(tpl.at * n)));
    const pos: Vec3 = [
      data.positions[idx * 3],
      data.positions[idx * 3 + 1],
      data.positions[idx * 3 + 2],
    ];
    return { id: tpl.id, kind: tpl.kind, pos, label: tpl.label, confidence: tpl.confidence };
  });
}
