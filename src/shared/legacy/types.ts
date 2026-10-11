// Primitive shapes used across every component. Deliberately tiny and
// dependency-free: the pure layer of shared must stay importable from a
// Node server, a browser viewer, and a Tauri app alike.

/** A point or offset, in whatever frame the surrounding type documents. */
export type Vec3 = [number, number, number];

/** Quaternion [x, y, z, w]. */
export type Quat = [number, number, number, number];
