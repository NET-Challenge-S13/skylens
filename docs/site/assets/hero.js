/* SkyLens guidebook hero.

   One city, cut into four flight segments, drawn at four densities. The left
   segments have already been refined to level 4; the right one has only just
   arrived at level 1. That stagger IS the delay pattern, and it is the reason
   the board shows something useful seconds after a drone passes instead of
   hours after the flight ends.

   Rules this file obeys:
     - it runs ONCE on load and then stops. No loop, no idle animation.
     - prefers-reduced-motion: reduce paints the final frame immediately.
     - a resize repaints the final frame; it never replays.

   Levels are drawn the way the reconstruction actually behaves: a sparse point
   cloud first, then more points, then a wireframe, then wireframe plus faces.
*/

(function () {
  'use strict';

  var canvas = document.getElementById('hero-scene');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');

  /* DRONE_TINTS from src/shared/viewer/config.ts. */
  var CYAN = '159,232,255';
  var AMBER = '255,210,127';
  var VIOLET = '199,159,255';

  var SEG_COUNT = 4;
  var TARGET = [4, 4, 3, 1];
  var DURATION = 1700;
  var STAGGER = 190;

  /* Deterministic skyline: the same city every load, in every browser. */
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Three rows of buildings. Further rows sit higher on the band, are shorter
     and dimmer, which is the whole depth cue; there is no real perspective. */
  var ROWS = [
    /* Counts are sized so every row reaches x = 1. Too few and the front row
       runs out around the third segment, leaving a hole under it. */
    { ground: 0.58, count: 64, hMin: 0.07, hMax: 0.26, depth: 5, alpha: 0.38 },
    { ground: 0.8, count: 48, hMin: 0.14, hMax: 0.48, depth: 8, alpha: 0.64 },
    /* The front row's base sits below the band so the city runs off the
       bottom edge instead of floating on an invisible shelf. */
    { ground: 1.06, count: 38, hMin: 0.2, hMax: 0.74, depth: 12, alpha: 0.95 }
  ];

  var buildings = [];

  (function buildCity() {
    var rnd = mulberry32(20250913);
    for (var r = 0; r < ROWS.length; r += 1) {
      var row = ROWS[r];
      var x = -0.02;
      for (var i = 0; i < row.count; i += 1) {
        var w = 0.008 + rnd() * 0.026;
        var h = row.hMin + rnd() * (row.hMax - row.hMin);
        if (x + w > 1.03) break;
        /* Points are revealed in a scrambled order so a half-resolved
           building looks sampled, not drawn top to bottom. */
        var order = [];
        for (var k = 0; k < 44; k += 1) order.push(k);
        for (var s = order.length - 1; s > 0; s -= 1) {
          var j = Math.floor(rnd() * (s + 1));
          var tmp = order[s];
          order[s] = order[j];
          order[j] = tmp;
        }
        buildings.push({
          x: x,
          w: w,
          h: h,
          ground: row.ground,
          depth: row.depth,
          alpha: row.alpha,
          seg: Math.min(SEG_COUNT - 1, Math.floor((x + w / 2) * SEG_COUNT)),
          order: order
        });
        x += w + 0.004 + rnd() * 0.016;
      }
    }
  })();

  /* Markers only exist where the scene has been reconstructed, so they are
     pinned to the two segments that reach level 4. Amber is a danger zone,
     violet is a person, the same as everywhere else on the site. */
  var MARKERS = [
    { x: 0.155, y: 0.6, r: 20, ink: AMBER, seg: 0 },
    { x: 0.4, y: 0.68, r: 16, ink: VIOLET, seg: 1 }
  ];

  var W = 0;
  var H = 0;
  var dpr = 1;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    var rect = canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(rect.width));
    H = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clamp01(v) {
    return v < 0 ? 0 : v > 1 ? 1 : v;
  }

  /* Outline of one building as a flat list of [x, y] pairs: the front face,
     then the roof plate offset up and to the right. */
  function outline(b) {
    var x0 = b.x * W;
    var x1 = (b.x + b.w) * W;
    var y1 = b.ground * H;
    var y0 = y1 - b.h * H;
    var d = b.depth;
    return {
      front: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
      roof: [[x0, y0], [x0 + d, y0 - d * 0.62], [x1 + d, y0 - d * 0.62], [x1, y0]],
      spine: [[x1, y0], [x1 + d, y0 - d * 0.62], [x1 + d, y1 - d * 0.62], [x1, y1]],
      x0: x0,
      x1: x1,
      y0: y0,
      y1: y1,
      d: d
    };
  }

  function polyPoints(o) {
    /* 44 sample positions spread over the front rectangle and the roof, used
       for the sparse point-cloud levels. */
    var pts = [];
    var i;
    for (i = 0; i < 14; i += 1) {
      pts.push([o.x0 + ((o.x1 - o.x0) * i) / 13, o.y0]);
    }
    for (i = 0; i < 10; i += 1) {
      pts.push([o.x0, o.y0 + ((o.y1 - o.y0) * i) / 9]);
      pts.push([o.x1, o.y0 + ((o.y1 - o.y0) * i) / 9]);
    }
    for (i = 0; i < 10; i += 1) {
      pts.push([o.x0 + o.d + ((o.x1 - o.x0) * i) / 9, o.y0 - o.d * 0.62]);
    }
    return pts;
  }

  function strokePoly(pts, close) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (var i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i][0], pts[i][1]);
    if (close) ctx.closePath();
    ctx.stroke();
  }

  function drawBuilding(b, level) {
    if (level <= 0.02) return;
    var o = outline(b);

    /* Point cloud: dense at level 1 to 2, fading out as the wireframe lands. */
    var pointT = clamp01(level / 2);
    var pointFade = 1 - clamp01(level - 2.4);
    if (pointFade > 0.01) {
      var pts = polyPoints(o);
      var n = Math.round(pts.length * (0.24 + 0.76 * pointT));
      ctx.fillStyle = 'rgba(' + CYAN + ',' + (0.66 * b.alpha * pointFade).toFixed(3) + ')';
      for (var i = 0; i < n; i += 1) {
        var p = pts[b.order[i % b.order.length] % pts.length];
        ctx.fillRect(Math.round(p[0]), Math.round(p[1]), 1.4, 1.4);
      }
    }

    /* Level 3 is the wireframe. Level 4 brightens the same lines rather than
       adding a different kind of mark, so the two levels differ by weight and
       not by vocabulary. Without this ramp a level-3 segment was
       indistinguishable from a level-4 one, which defeats the whole point. */
    var solid = clamp01(level - 3);
    var wire = clamp01(level - 2);
    if (wire > 0.01) {
      ctx.lineWidth = 1;
      ctx.strokeStyle =
        'rgba(' + CYAN + ',' + ((0.4 + 0.42 * solid) * b.alpha * wire).toFixed(3) + ')';
      strokePoly(o.front, true);
      strokePoly(o.roof, true);
      strokePoly(o.spine, true);
    }

    /* Faces, floor lines and a lit roof edge: the settled state. */
    if (solid > 0.01) {
      ctx.fillStyle = 'rgba(' + CYAN + ',' + (0.17 * b.alpha * solid).toFixed(3) + ')';
      ctx.beginPath();
      ctx.moveTo(o.roof[0][0], o.roof[0][1]);
      for (var k = 1; k < o.roof.length; k += 1) ctx.lineTo(o.roof[k][0], o.roof[k][1]);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = 'rgba(' + CYAN + ',' + (0.085 * b.alpha * solid).toFixed(3) + ')';
      ctx.fillRect(o.x0, o.y0, o.x1 - o.x0, o.y1 - o.y0);

      /* Floor lines are the detail that only a refined segment has. */
      var floors = Math.max(1, Math.round((o.y1 - o.y0) / 26));
      ctx.strokeStyle = 'rgba(' + CYAN + ',' + (0.3 * b.alpha * solid).toFixed(3) + ')';
      ctx.beginPath();
      for (var f = 1; f < floors; f += 1) {
        var fy = Math.round(o.y0 + ((o.y1 - o.y0) * f) / floors) + 0.5;
        ctx.moveTo(o.x0, fy);
        ctx.lineTo(o.x1, fy);
      }
      ctx.stroke();

      ctx.strokeStyle = 'rgba(' + CYAN + ',' + (0.95 * b.alpha * solid).toFixed(3) + ')';
      ctx.beginPath();
      ctx.moveTo(o.x0 + o.d, o.y0 - o.d * 0.62);
      ctx.lineTo(o.x1 + o.d, o.y0 - o.d * 0.62);
      ctx.stroke();
    }
  }

  /* The camera frustum shape from the control tower, used as a map pin. */
  function drawMarker(m, t) {
    if (t <= 0.01) return;
    var cx = m.x * W;
    var cy = m.y * H;
    var r = m.r;
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(' + m.ink + ',' + (0.85 * t).toFixed(3) + ')';
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 0.62);
    ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r * 0.62);
    ctx.lineTo(cx - r, cy);
    ctx.closePath();
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, cy - r * 1.5);
    ctx.lineTo(cx + r, cy);
    ctx.moveTo(cx, cy - r * 1.5);
    ctx.lineTo(cx - r, cy);
    ctx.moveTo(cx, cy - r * 1.5);
    ctx.lineTo(cx, cy + r * 0.62);
    ctx.stroke();
    ctx.fillStyle = 'rgba(' + m.ink + ',' + t.toFixed(3) + ')';
    ctx.fillRect(cx - 1.5, cy - r * 1.5 - 1.5, 3, 3);
  }

  /* Ground lines: a couple of faint roads so the city sits on something. */
  function drawGround(levels) {
    for (var r = 0; r < ROWS.length; r += 1) {
      var y = Math.round(ROWS[r].ground * H) + 0.5;
      for (var s = 0; s < SEG_COUNT; s += 1) {
        var t = clamp01(levels[s] - 1.2);
        if (t <= 0.01) continue;
        ctx.strokeStyle = 'rgba(' + CYAN + ',' + (0.12 * ROWS[r].alpha * t).toFixed(3) + ')';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo((s / SEG_COUNT) * W, y);
        ctx.lineTo(((s + 1) / SEG_COUNT) * W, y);
        ctx.stroke();
      }
    }
  }

  function render(levels) {
    ctx.clearRect(0, 0, W, H);
    drawGround(levels);
    for (var i = 0; i < buildings.length; i += 1) {
      var b = buildings[i];
      drawBuilding(b, levels[b.seg]);
    }
    for (var m = 0; m < MARKERS.length; m += 1) {
      drawMarker(MARKERS[m], clamp01((levels[MARKERS[m].seg] - 3.4) / 0.6));
    }
  }

  function finalLevels() {
    return TARGET.slice();
  }

  function easeOut(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  var finished = false;

  function paintFinal() {
    resize();
    render(finalLevels());
    finished = true;
  }

  var reduce =
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (reduce) {
    paintFinal();
  } else {
    resize();
    var start = null;
    var step = function (now) {
      if (start === null) start = now;
      var elapsed = now - start;
      var levels = [];
      var done = true;
      for (var s = 0; s < SEG_COUNT; s += 1) {
        var t = clamp01((elapsed - s * STAGGER) / DURATION);
        if (t < 1) done = false;
        levels.push(TARGET[s] * easeOut(t));
      }
      render(levels);
      if (done) {
        finished = true;
        return;
      }
      window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
  }

  /* A resize repaints the settled frame. It never replays the resolve. */
  var pending = 0;
  window.addEventListener('resize', function () {
    window.clearTimeout(pending);
    pending = window.setTimeout(function () {
      if (!finished) return;
      resize();
      render(finalLevels());
    }, 120);
  });
})();
