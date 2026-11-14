const { chromium } = require('playwright');
const R = 'run:2026-11-12-inkling-2-small-';
const M64 = 'tinker:thinkingmachines/Inkling-2-Small', M256 = 'tinker:thinkingmachines/Inkling-2-Small:peft:262144';
const shots = [
  ['graph-0313-262k-base', 'gnlw-2025-03-13-0706-180m-real-0', `${R}256k-max-eval45:${M256}@max`],
  ['graph-0313-262k-rl', 'gnlw-2025-03-13-0706-180m-real-0', `${R}256k-max-rl-b-eval45:${M256}#rl-b@max`],
  ['graph-0313-64k-base', 'gnlw-2025-03-13-0706-180m-real-0', `${R}base-eval45:${M64}@high`],
  ['graph-0313-64k-rl', 'gnlw-2025-03-13-0706-180m-real-0', `${R}rl-a-eval45:${M64}#rl-a@high`],
  ['graph-0216-262k-base', 'gnlw-2025-02-16-1700-210m-compound-0', `${R}256k-max-eval45:${M256}@max`],
  ['graph-0216-262k-rl', 'gnlw-2025-02-16-1700-210m-compound-0', `${R}256k-max-rl-b-eval45:${M256}#rl-b@max`],
  ['graph-0531-262k-base', 'gnlw-2025-05-31-0900-180m-compound-0', `${R}256k-max-eval45:${M256}@max`],
  ['graph-0531-262k-rl', 'gnlw-2025-05-31-0900-180m-compound-0', `${R}256k-max-rl-b-eval45:${M256}#rl-b@max`],
];
(async () => {
  const b = await chromium.launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: '127.0.0.1,localhost' } });
  const pg = await b.newPage({ viewport: { width: 1052, height: 900 }, deviceScaleFactor: 2, ignoreHTTPSErrors: true });
  for (const [name, task, plan] of shots) {
    await pg.goto(`http://127.0.0.1:8031/viewer/?task=${task}&plan=${encodeURIComponent(plan)}&figure=1`);
    await pg.waitForTimeout(3500); await pg.addStyleTag({ content: '.card ol{display:none!important}' }); await pg.waitForTimeout(300);
    const card = await pg.$('.card');
    await (card || pg).screenshot({ path: `blogmedia/${name}.png` });
    console.log(name, 'ok');
  }
  await b.close();
})();
