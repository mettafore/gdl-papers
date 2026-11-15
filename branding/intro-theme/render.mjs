// Renders theme.html frame by frame with headless Chromium.
// usage: node render.mjs <outDir> <fps> [startFrame endFrame] [query]
import { createRequire } from 'module';
const { chromium } = createRequire(import.meta.url)('playwright');
import { pathToFileURL } from 'url';
import fs from 'fs'; import path from 'path';
const [out, fpsS, s, e, query = ''] = process.argv.slice(2);
const fps = +fpsS; fs.mkdirSync(out, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(path.resolve(import.meta.dirname, process.env.PAGE || 'theme.html')).href + (query ? '?' + query : ''));
await page.evaluate(() => window.ready);
const total = Math.round(await page.evaluate(() => window.DUR) * fps);
const a = s ? +s : 0, b = e ? Math.min(+e, total) : total;
for (let f = a; f < b; f++) {
  await page.evaluate(t => window.renderFrame(t), f / fps);
  await page.screenshot({ path: path.join(out, `f${String(f).padStart(5, '0')}.png`) });
}
await browser.close();
