// node shot3d.js '<json spec>': {name, task, plans:{key:planId}, train, cam, from, to, step, w, h, settle}
// Writes shots/<name>/<key>-NNNN.png and shots/<name>/meta.json (clock per frame, plan label and reward per key).
// Frames already on disk are kept, so a run cut off by a restart resumes where it stopped.
const { chromium } = require('playwright');
const fs = require('fs');
const spec = JSON.parse(process.argv[2]);
const hm = (s) => { const [h, m] = s.split(':').map(Number); return h * 60 + m; };
(async () => {
  const dir = `shots/${spec.name}`; fs.mkdirSync(dir, { recursive: true });
  const b = await chromium.launch({ args: ['--use-angle=swiftshader'], proxy: { server: process.env.HTTPS_PROXY, bypass: '127.0.0.1,localhost' } });
  const pages = {}, meta = { spec, frames: [], plans: {} };
  for (const [k, plan] of Object.entries(spec.plans)) {
    const pg = await b.newPage({ viewport: { width: spec.w || 1280, height: spec.h || 720 }, ignoreHTTPSErrors: true });
    await pg.goto(`http://127.0.0.1:8031/twin/?task=${spec.task}&plan=${encodeURIComponent(plan)}`);
    await pg.waitForSelector('#loading.done', { timeout: 240000 });
    await pg.addStyleTag({ content: '#bar,.panel,#toasts{display:none!important}' });
    meta.plans[k] = await pg.evaluate(({ train, cam }) => {
      const r = twin.state.R.runs.find((x) => x.train === train);
      if (r) { twin.select(r.unit, false); twin.setCam(cam); }
      const lab = document.querySelector('#plan').selectedOptions[0].textContent;
      return { label: lab, reward: twin.state.R.reward, feasible: twin.state.R.feasible, unit: r ? r.unit : null };
    }, { train: spec.train, cam: spec.cam });
    pages[k] = pg;
  }
  const a = hm(spec.from), z = hm(spec.to), step = spec.step || 0.25;
  let i = 0;
  for (let t = a; t <= z + 1e-9; t += step, i++) {
    meta.frames.push(t);
    const files = Object.keys(pages).map((k) => `${dir}/${k}-${String(i).padStart(4, '0')}.png`);
    if (files.every((f) => fs.existsSync(f))) continue;     // kept from a run that was cut off: resume after it
    for (const pg of Object.values(pages)) await pg.evaluate((t) => twin.seek(t), t);
    await Object.values(pages)[0].waitForTimeout(spec.settle || 450);
    for (const [k, pg] of Object.entries(pages)) await pg.screenshot({ path: `${dir}/${k}-${String(i).padStart(4, '0')}.png` });
  }
  fs.writeFileSync(`${dir}/meta.json`, JSON.stringify(meta, null, 1));
  console.log(spec.name, i, 'frames', JSON.stringify(meta.plans));
  await b.close();
})();
