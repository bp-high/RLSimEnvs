// Inline each page's module scripts, bundled with their static imports, into the HTML.
//
//   node flatten.cjs <built site> <template app folder (has node_modules/esbuild)>
//
// A private Space is served to its iframe with the access token in the page's address only, and the Space lets a
// file through when the request's Referer is that address. A script imported by another script is requested with
// the importing script as Referer and refused (401), which took the theme toggle and the table of contents down.
// Inlined, every remaining request (stylesheets, data, dynamic imports) comes from the page itself.
const fs = require("fs");
const path = require("path");
const esbuild = require(require.resolve("esbuild", { paths: [process.argv[3]] }));

const site = path.resolve(process.argv[2]);
const TAG = /<script type="module" src="(\/[^"]+\.js)"><\/script>/g;

// Dynamic imports stay separate files, addressed from the site root so they resolve from an inline script too.
const keepDynamic = {
  name: "keep-dynamic",
  setup(b) {
    b.onResolve({ filter: /.*/ }, (a) =>
      a.kind === "dynamic-import"
        ? { path: "/" + path.relative(site, path.resolve(a.resolveDir, a.path)).split(path.sep).join("/"), external: true }
        : undefined);
  },
};

async function bundle(src) {
  const out = await esbuild.build({
    entryPoints: [path.join(site, src)], bundle: true, format: "esm", minify: true, write: false,
    plugins: [keepDynamic], logLevel: "error",
  });
  return out.outputFiles[0].text.replace(/<\/script/gi, "<\\/script");
}

function htmlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    return e.isDirectory() ? htmlFiles(p) : e.name.endsWith(".html") ? [p] : [];
  });
}

(async () => {
  const cache = new Map();
  let n = 0;
  for (const f of htmlFiles(site)) {
    const html = fs.readFileSync(f, "utf8");
    for (const [, src] of html.matchAll(TAG)) if (!cache.has(src)) cache.set(src, await bundle(src));
    const next = html.replace(TAG, (_, src) => (n++, `<script type="module">${cache.get(src)}</script>`));
    if (next !== html) fs.writeFileSync(f, next);
  }
  console.log(`inlined ${n} module script tags (${cache.size} scripts)`);
})().catch((e) => { console.error(e); process.exit(1); });
