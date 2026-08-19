// Match two pictures of the same ground and report the offset in metres.
//
// Used by the checks that compare the planner map against the 3D view. Both
// images must cover the same ground span across their full width; anything that
// differs after that — a shift, a scale — is the answer being looked for.
//
// The work happens in a browser page because that is where an image decoder and
// a canvas already exist; no image library is needed on the node side.

/**
 * @param page      a Playwright page (any page — only canvas APIs are used)
 * @param aPng      Buffer: the reference image
 * @param bPng      Buffer: the image being checked against it
 * @param spanM     ground width both images cover, in metres
 * @returns {{score:number, offsetM:{east:number,north:number}, scale:number}}
 */
export async function matchImages(page, aPng, bPng, spanM) {
  return page.evaluate(
    async ([aB64, bB64, span]) => {
      const load = async (b64) => {
        const res = await fetch(`data:image/png;base64,${b64}`);
        return createImageBitmap(await res.blob());
      };
      /** Grayscale square, `n` px across, resampled from a bitmap. */
      const gray = (bmp, n) => {
        const c = new OffscreenCanvas(n, n);
        const g = c.getContext('2d');
        g.drawImage(bmp, 0, 0, n, n);
        const d = g.getImageData(0, 0, n, n).data;
        const out = new Float32Array(n * n);
        for (let i = 0; i < n * n; i++) {
          out[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
        }
        return out;
      };
      /** Normalised cross-correlation of a template against a window. */
      const ncc = (t, tN, s, sN, ox, oy, scale) => {
        let sumT = 0;
        let sumS = 0;
        let n = 0;
        const vals = [];
        for (let y = 0; y < tN; y += 2) {
          for (let x = 0; x < tN; x += 2) {
            const sx = Math.round((x - tN / 2) * scale + sN / 2 + ox);
            const sy = Math.round((y - tN / 2) * scale + sN / 2 + oy);
            if (sx < 0 || sy < 0 || sx >= sN || sy >= sN) continue;
            vals.push(t[y * tN + x], s[sy * sN + sx]);
            sumT += t[y * tN + x];
            sumS += s[sy * sN + sx];
            n++;
          }
        }
        if (n < 200) return -1;
        const mT = sumT / n;
        const mS = sumS / n;
        let num = 0;
        let dT = 0;
        let dS = 0;
        for (let i = 0; i < vals.length; i += 2) {
          const a = vals[i] - mT;
          const b = vals[i + 1] - mS;
          num += a * b;
          dT += a * a;
          dS += b * b;
        }
        return dT > 0 && dS > 0 ? num / Math.sqrt(dT * dS) : -1;
      };

      const N = 150;
      const a = gray(await load(aB64), N);
      const b = gray(await load(bB64), N);

      // Template: the middle of the reference. The edges of a 3D capture carry
      // HUD panels, and the middle is the part an operator plans in anyway.
      const tN = 80;
      const tpl = new Float32Array(tN * tN);
      const off = Math.floor((N - tN) / 2);
      for (let y = 0; y < tN; y++) {
        for (let x = 0; x < tN; x++) tpl[y * tN + x] = a[(y + off) * N + (x + off)];
      }

      let best = { score: -2, dx: 0, dy: 0, scale: 1 };
      for (let scale = 0.6; scale <= 1.65; scale += 0.05) {
        for (let dy = -30; dy <= 30; dy++) {
          for (let dx = -30; dx <= 30; dx++) {
            const score = ncc(tpl, tN, b, N, dx, dy, scale);
            if (score > best.score) best = { score, dx, dy, scale };
          }
        }
      }
      const mPerPx = span / N;
      return {
        score: Number(best.score.toFixed(3)),
        offsetM: {
          east: Number((best.dx * mPerPx).toFixed(1)),
          north: Number((-best.dy * mPerPx).toFixed(1)),
        },
        scale: Number(best.scale.toFixed(2)),
      };
    },
    [aPng.toString('base64'), bPng.toString('base64'), spanM],
  );
}
