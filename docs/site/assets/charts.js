/* Guidebook charts. Plain SVG, no library: each <figure data-chart="..."> gets its
   plot drawn into its .chart__plot, redrawn on resize, with one shared tooltip.
   The numbers live here; the tables next to each chart carry the same numbers
   for anyone who does not hover. */

(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var INK = {
    cyan: '#9fe8ff',
    amber: '#ffd27f',
    slate: '#93a5c2',
    grid: 'rgba(126, 165, 219, 0.14)',
    axis: 'rgba(147, 183, 232, 0.42)',
    text: '#8092ad',
    strong: '#e9eef7',
    void: '#05070a'
  };

  /* ---- data ------------------------------------------------------------ */

  // models/skylens/RESULTS.md, experiment 2: ALIKED 540 frames, one 30,000-step run.
  var STEPS = [
    { step: 250, psnr: 16.09, ssim: 0.838, lpips: 0.531, gauss: 12708, note: '형상 윤곽만' },
    { step: 500, psnr: 17.23, ssim: 0.851, lpips: 0.51, gauss: 12708, note: '' },
    { step: 1000, psnr: 18.47, ssim: 0.86, lpips: 0.479, gauss: 19368, note: '공간 구조 인식 가능' },
    { step: 2000, psnr: 19.8, ssim: 0.875, lpips: 0.424, gauss: 69919, note: '' },
    { step: 3500, psnr: 20.48, ssim: 0.881, lpips: 0.392, gauss: 146160, note: '표면 채워짐' },
    { step: 5000, psnr: 20.91, ssim: 0.886, lpips: 0.376, gauss: 221076, note: '' },
    { step: 7000, psnr: 21.48, ssim: 0.891, lpips: 0.357, gauss: 316057, note: '실용 품질' },
    { step: 10000, psnr: 22.02, ssim: 0.895, lpips: 0.339, gauss: 454965, note: '' },
    { step: 15000, psnr: 22.41, ssim: 0.897, lpips: 0.327, gauss: 653353, note: '밀도 제어 종료' },
    { step: 20000, psnr: 23.27, ssim: 0.904, lpips: 0.309, gauss: 653353, note: '' },
    { step: 30000, psnr: 23.62, ssim: 0.907, lpips: 0.301, gauss: 653353, note: '최종' }
  ];

  // Same file, stage times. Seconds.
  var STAGES = {
    series: [
      { key: 'sift', name: '360장 · SIFT', color: INK.slate },
      { key: 'aliked', name: '540장 · ALIKED', color: INK.cyan }
    ],
    rows: [
      { name: '특징점 추출', sift: 26, aliked: 35 },
      { name: '전수 매칭', sift: 507, aliked: 2535 },
      { name: '매퍼 (SfM)', sift: 797, aliked: 3254 }
    ]
  };

  // Whole pipeline on the KOREN deployment: three drones, 53 s of footage each, 2.5 Mbps SRT.
  // `len` is the measured duration of each step.
  var TIMELINE = {
    end: 324,
    bars: [
      { name: '첫 프레임 묶음 도착', from: 0, to: 22.1, len: 22.1, detail: '드론 3대 SRT 송출 시작부터 판교 V100에 첫 카메라 3장 묶음이 닿기까지' },
      { name: '첫 구역 복원', from: 22.1, to: 39.2, len: 17.1, detail: '첫 구역을 복원해 현황판에 첫 데이터를 띄우기까지' },
      { name: '프레임 수신', from: 22.1, to: 92, len: 70.4, detail: '첫 프레임부터 마지막 프레임까지 받는 구간' },
      { name: '남은 구역 복원', from: 92, to: 324, len: 231.8, detail: '마지막 프레임 이후 나머지 구역을 복원해 보내기까지' }
    ],
    marks: [
      { at: 39.2, name: '현황판 첫 표시', short: '첫 표시' },
      { at: 324, name: '마지막 데이터 도착', short: '마지막' }
    ]
  };

  /* ---- helpers --------------------------------------------------------- */

  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function text(parent, x, y, s, attrs) {
    var t = el('text', Object.assign({ x: x, y: y, fill: INK.text, 'font-size': 11.5 }, attrs || {}), parent);
    t.textContent = s;
    return t;
  }

  function grouped(n) {
    return Math.round(n).toLocaleString('ko-KR');
  }

  function clock(sec) {
    var m = Math.floor(sec / 60);
    var s = Math.round(sec - m * 60);
    if (s === 60) { m += 1; s = 0; }
    return m > 0 ? m + '분 ' + (s < 10 ? '0' : '') + s + '초' : s + '초';
  }

  /* One tooltip per figure. Rows are [value, label] pairs; the value leads. */
  function tooltip(fig) {
    var tip = document.createElement('div');
    tip.className = 'chart__tip';
    tip.setAttribute('role', 'status');
    tip.hidden = true;
    fig.querySelector('.chart__plot').appendChild(tip);
    return {
      show: function (x, y, title, rows) {
        tip.textContent = '';
        var h = document.createElement('p');
        h.className = 'chart__tip-title';
        h.textContent = title;
        tip.appendChild(h);
        rows.forEach(function (r) {
          var p = document.createElement('p');
          var b = document.createElement('b');
          b.textContent = r[0];
          var span = document.createElement('span');
          span.textContent = r[1];
          p.appendChild(b);
          p.appendChild(span);
          tip.appendChild(p);
        });
        tip.hidden = false;
        var box = tip.parentNode.getBoundingClientRect();
        var w = tip.offsetWidth;
        var left = x + 14 + w > box.width ? x - 14 - w : x + 14;
        tip.style.left = Math.max(0, left) + 'px';
        tip.style.top = Math.max(0, y - tip.offsetHeight / 2) + 'px';
      },
      hide: function () { tip.hidden = true; }
    };
  }

  function frame(fig, height) {
    var plot = fig.querySelector('.chart__plot');
    var old = plot.querySelector('svg');
    if (old) old.remove();
    var w = Math.max(300, plot.clientWidth);
    var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + height, width: w, height: height, role: 'img' });
    svg.setAttribute('aria-label', fig.getAttribute('data-label') || '');
    plot.insertBefore(svg, plot.firstChild);
    return { svg: svg, w: w, h: height };
  }

  /* ---- PSNR over training steps (log x) --------------------------------- */

  function psnr(fig) {
    var f = frame(fig, 300);
    var narrow = f.w < 520;
    var m = { l: 44, r: narrow ? 18 : 30, t: 22, b: 40 };
    var iw = f.w - m.l - m.r;
    var ih = f.h - m.t - m.b;
    var lx0 = Math.log10(200);
    var lx1 = Math.log10(36000);
    var y0 = 15;
    var y1 = 24.5;
    var X = function (s) { return m.l + (Math.log10(s) - lx0) / (lx1 - lx0) * iw; };
    var Y = function (v) { return m.t + (1 - (v - y0) / (y1 - y0)) * ih; };
    var g = el('g', {}, f.svg);

    [16, 18, 20, 22, 24].forEach(function (v) {
      el('line', { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v), stroke: INK.grid }, g);
      text(g, m.l - 8, Y(v) + 4, String(v), { 'text-anchor': 'end' });
    });
    [250, 1000, 3500, 10000, 30000].forEach(function (s) {
      text(g, X(s), m.t + ih + 18, grouped(s), { 'text-anchor': 'middle' });
    });
    el('line', { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: INK.axis }, g);
    text(g, m.l + iw, m.t + ih + 34, '학습 스텝 (로그 축)', { 'text-anchor': 'end' });
    text(g, m.l - 36, m.t - 8, 'PSNR (dB)', {});

    var d = STEPS.map(function (p, i) { return (i ? 'L' : 'M') + X(p.step) + ' ' + Y(p.psnr); }).join(' ');
    el('path', { d: d, fill: 'none', stroke: INK.cyan, 'stroke-width': 2, 'stroke-linejoin': 'round' }, g);
    STEPS.forEach(function (p) {
      el('circle', { cx: X(p.step), cy: Y(p.psnr), r: 4, fill: INK.cyan, stroke: INK.void, 'stroke-width': 2 }, g);
    });

    // Two direct labels only: where the scene becomes readable, and the end.
    var a = STEPS[2];
    var z = STEPS[STEPS.length - 1];
    text(g, X(a.step) + 8, Y(a.psnr) + 18, '1,000스텝 · 구조가 보여요', { fill: INK.strong });
    text(g, X(z.step) - 8, Y(z.psnr) - 12, '최종 23.62', { fill: INK.strong, 'text-anchor': 'end' });

    var hair = el('line', { y1: m.t, y2: m.t + ih, stroke: INK.axis, 'stroke-dasharray': '3 3', visibility: 'hidden' }, g);
    var dot = el('circle', { r: 6, fill: 'none', stroke: INK.strong, 'stroke-width': 2, visibility: 'hidden' }, g);
    var tip = tooltip(fig);
    var hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent', tabindex: 0 }, g);

    function at(i) {
      var p = STEPS[i];
      var x = X(p.step);
      hair.setAttribute('x1', x);
      hair.setAttribute('x2', x);
      hair.setAttribute('visibility', 'visible');
      dot.setAttribute('cx', x);
      dot.setAttribute('cy', Y(p.psnr));
      dot.setAttribute('visibility', 'visible');
      var rows = [
        [p.psnr.toFixed(2) + ' dB', 'PSNR'],
        [p.ssim.toFixed(3), 'SSIM'],
        [p.lpips.toFixed(3), 'LPIPS'],
        [grouped(p.gauss), '가우시안']
      ];
      if (p.note) rows.push(['', p.note]);
      tip.show(x, Y(p.psnr), grouped(p.step) + ' 스텝', rows);
    }
    var cur = STEPS.length - 1;
    hit.addEventListener('pointermove', function (e) {
      var r = f.svg.getBoundingClientRect();
      var x = (e.clientX - r.left) * (f.w / r.width);
      var best = 0;
      STEPS.forEach(function (p, i) {
        if (Math.abs(X(p.step) - x) < Math.abs(X(STEPS[best].step) - x)) best = i;
      });
      cur = best;
      at(best);
    });
    hit.addEventListener('focus', function () { at(cur); });
    hit.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight') cur = Math.min(STEPS.length - 1, cur + 1);
      else if (e.key === 'ArrowLeft') cur = Math.max(0, cur - 1);
      else return;
      e.preventDefault();
      at(cur);
    });
    function off() {
      hair.setAttribute('visibility', 'hidden');
      dot.setAttribute('visibility', 'hidden');
      tip.hide();
    }
    hit.addEventListener('pointerleave', off);
    hit.addEventListener('blur', off);
  }

  /* ---- stage times, grouped horizontal bars ---------------------------- */

  function stages(fig) {
    var rows = STAGES.rows;
    var series = STAGES.series;
    var band = 54;
    var f = frame(fig, 34 + rows.length * band + 30);
    var narrow = f.w < 520;
    var m = { l: narrow ? 78 : 96, r: narrow ? 62 : 80, t: 30, b: 30 };
    var iw = f.w - m.l - m.r;
    var max = 3300;
    var X = function (s) { return m.l + s / max * iw; };
    var g = el('g', {}, f.svg);
    var tip = tooltip(fig);

    // Legend, above the plot.
    var lx = m.l;
    series.forEach(function (s) {
      el('rect', { x: lx, y: 8, width: 12, height: 10, rx: 2, fill: s.color }, g);
      var t = text(g, lx + 18, 17, s.name, { fill: INK.strong });
      lx += 18 + t.getComputedTextLength() + 22;
    });

    [0, 1200, 2400].forEach(function (s) {
      el('line', { x1: X(s), x2: X(s), y1: m.t, y2: m.t + rows.length * band, stroke: INK.grid }, g);
      text(g, X(s), m.t + rows.length * band + 18, s ? (s / 60) + '분' : '0', { 'text-anchor': 'middle' });
    });

    rows.forEach(function (row, i) {
      var y = m.t + i * band + 8;
      text(g, m.l - 10, y + 20, row.name, { 'text-anchor': 'end', fill: INK.strong });
      series.forEach(function (s, j) {
        var v = row[s.key];
        var by = y + j * 18;
        var bw = Math.max(3, X(v) - m.l);
        var bar = el('rect', { x: m.l, y: by, width: bw, height: 14, rx: 2, fill: s.color, tabindex: 0 }, g);
        text(g, m.l + bw + 6, by + 11, clock(v), {});
        var show = function () {
          bar.setAttribute('opacity', 0.75);
          tip.show(m.l + bw, by + 7, row.name, [[clock(v), s.name]]);
        };
        var hide = function () { bar.setAttribute('opacity', 1); tip.hide(); };
        bar.addEventListener('pointerenter', show);
        bar.addEventListener('focus', show);
        bar.addEventListener('pointerleave', hide);
        bar.addEventListener('blur', hide);
      });
    });
  }

  /* ---- pipeline timeline (gantt) --------------------------------------- */

  function timeline(fig) {
    var bars = TIMELINE.bars;
    var band = 40;
    var f = frame(fig, 40 + bars.length * band + 44);
    var narrow = f.w < 560;
    var m = { l: narrow ? 122 : 136, r: 24, t: 34, b: 44 };
    var iw = f.w - m.l - m.r;
    var X = function (s) { return m.l + s / 340 * iw; };
    var g = el('g', {}, f.svg);
    var tip = tooltip(fig);
    var bottom = m.t + bars.length * band;

    [0, 60, 120, 180, 240, 300].forEach(function (s) {
      el('line', { x1: X(s), x2: X(s), y1: m.t - 6, y2: bottom, stroke: INK.grid }, g);
      text(g, X(s), bottom + 18, s + '초', { 'text-anchor': 'middle' });
    });
    text(g, m.l + iw, bottom + 36, '송출 시작부터 흐른 시간', { 'text-anchor': 'end' });

    bars.forEach(function (b, i) {
      var y = m.t + i * band + 10;
      text(g, m.l - 10, y + 13, b.name, { 'text-anchor': 'end', fill: INK.strong });
      var w = Math.max(3, X(b.to) - X(b.from));
      var bar = el('rect', { x: X(b.from), y: y, width: w, height: 18, rx: 2, fill: INK.cyan, tabindex: 0 }, g);
      // The measured durations, not to - from: the table's times are rounded (92 s).
      var len = b.len;
      var lbl = len.toFixed(1) + '초';
      if (w > 56) text(g, X(b.from) + 8, y + 13, lbl, { fill: INK.void, 'font-weight': 600 });
      else text(g, X(b.to) + 6, y + 13, lbl, {});
      var show = function () {
        bar.setAttribute('opacity', 0.75);
        tip.show(X(b.to), y + 9, b.name, [
          [len.toFixed(1) + '초', '걸린 시간'],
          [b.from + '초 → ' + b.to + '초', '구간'],
          ['', b.detail]
        ]);
      };
      var hide = function () { bar.setAttribute('opacity', 1); tip.hide(); };
      bar.addEventListener('pointerenter', show);
      bar.addEventListener('focus', show);
      bar.addEventListener('pointerleave', hide);
      bar.addEventListener('blur', hide);
    });

    // Milestones: what the commander sees. Amber is the board's ink.
    TIMELINE.marks.forEach(function (k, i) {
      var x = X(k.at);
      el('line', { x1: x, x2: x, y1: m.t - 10, y2: bottom, stroke: INK.amber, 'stroke-dasharray': '4 3' }, g);
      el('path', { d: 'M' + x + ' ' + (m.t - 18) + ' l5 5 l-5 5 l-5 -5 z', fill: INK.amber }, g);
      var right = i === TIMELINE.marks.length - 1;
      text(g, right ? x - 9 : x + 9, m.t - 9, (narrow ? k.short : k.name) + ' ' + k.at + '초', {
        fill: INK.strong,
        'text-anchor': right ? 'end' : 'start'
      });
    });
  }

  /* ---- wiring ------------------------------------------------------------ */

  var DRAW = { psnr: psnr, stages: stages, timeline: timeline };
  var figs = document.querySelectorAll('figure[data-chart]');

  function drawAll() {
    for (var i = 0; i < figs.length; i += 1) {
      var fn = DRAW[figs[i].getAttribute('data-chart')];
      if (fn) {
        var tipOld = figs[i].querySelector('.chart__tip');
        if (tipOld) tipOld.remove();
        fn(figs[i]);
      }
    }
  }

  var last = 0;
  window.addEventListener('resize', function () {
    var w = window.innerWidth;
    if (w === last) return;
    last = w;
    window.requestAnimationFrame(drawAll);
  });
  drawAll();
})();
