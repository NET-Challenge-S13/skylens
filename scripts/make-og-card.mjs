// 가이드북의 링크 미리보기 이미지(docs/site/assets/og.png)를 다시 만든다.
//
// 카카오톡·슬랙·트위터가 og:image 로 읽는 카드다. 홈 화면을 그대로 1200x630 으로
// 찍되, 제목과 한 줄 소개가 잘리지 않도록 위치를 맞춘다. 히어로 띠는 위쪽이 잘려도
// 되지만 글자는 잘리면 안 된다.
//
// 실행:  "C:/Program Files/nodejs/node.exe" scripts/make-og-card.mjs
// 홈 화면을 고치면 이 스크립트를 다시 돌려서 카드도 같이 갱신한다.

import { chromium } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PAGE = 'file:///' + path.join(ROOT, 'docs/site/index.html').replaceAll('\\', '/');
const OUT = path.join(ROOT, 'docs/site/assets/og.png');

const WIDTH = 1200;
const HEIGHT = 630; // 카카오톡 권장 비율(1.91:1)
const BOTTOM_MARGIN = 46;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });

await page.goto(PAGE);
await page.waitForTimeout(2500); // 히어로 해상 애니메이션이 끝날 때까지

const box = await page.evaluate(() => {
  // 카드에 들어가면 안 되는 화면 요소를 뺀다.
  document.querySelector('.skip-link')?.remove();
  document.querySelector('.site-header')?.remove();
  document.querySelectorAll('.btn').forEach((el) => el.remove());
  document.documentElement.style.overflow = 'hidden';

  const h1 = document.querySelector('h1');
  const lede = document.querySelector('.lede') ?? h1;
  return {
    top: h1.getBoundingClientRect().top + window.scrollY,
    bottom: lede.getBoundingClientRect().bottom + window.scrollY,
  };
});

// 글자 블록을 아래쪽에 온전히 놓고, 남는 위 공간을 히어로 띠로 채운다.
const textHeight = box.bottom - box.top;
const scrollTo = Math.max(0, Math.round(box.top - (HEIGHT - textHeight - BOTTOM_MARGIN)));
await page.evaluate((y) => window.scrollTo(0, y), scrollTo);
await page.waitForTimeout(250);

await page.screenshot({ path: OUT });
await browser.close();

console.log(`${OUT} (${WIDTH}x${HEIGHT}), 글자 블록 ${Math.round(textHeight)}px, scrollY ${scrollTo}`);
