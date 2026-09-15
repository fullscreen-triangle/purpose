import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage();
const errors = [];
page.on('console', msg => { if (msg.type() === 'error') errors.push(msg.text()); });
page.on('pageerror', err => errors.push('pageerror: ' + err.message));

await page.goto('http://localhost:4173/', { waitUntil: 'networkidle' });
await page.waitForTimeout(500);

const svgCount = await page.locator('svg').count();
const tableRows = await page.locator('#exp-table-body tr').count();
const heroTitle = await page.locator('h1.title').textContent();
const sectionCount = await page.locator('section').count();

console.log('SVG count:', svgCount);
console.log('Experiment table rows:', tableRows);
console.log('Hero title:', JSON.stringify(heroTitle));
console.log('Section count:', sectionCount);
console.log('Console/page errors:', errors.length ? errors : 'none');

await page.screenshot({ path: 'C:\\Users\\kunda\\AppData\\Local\\Temp\\claude\\c--Users-kunda-Documents-semantics-purpose\\51525d69-40c2-4c27-8cfc-69faaddbd447\\scratchpad\\screenshot.png', fullPage: false });

await browser.close();
process.exit(errors.length ? 1 : 0);
