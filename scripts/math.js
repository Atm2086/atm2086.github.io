'use strict';

const fs = require('node:fs');
const path = require('node:path');
const markedKatex = require('marked-katex-extension');

// Tokenize TeX before Markdown can consume its backslashes or underscores.
const { extensions } = markedKatex({
  nonStandard: true, // Also support formulas adjacent to Chinese text.
  throwOnError: false,
  strict: 'ignore' // Existing notes include Chinese labels inside formulas.
});

// A display equation can follow a paragraph without an intervening blank line.
extensions.find(extension => extension.level === 'block').start = src => {
  const match = /\n\${1,2}\n/.exec(src);
  return match ? match.index + 1 : undefined;
};

hexo.extend.filter.register('marked:extensions', function (registered) {
  registered.push(...extensions);
});

// Publish CSS and its matching fonts locally; no browser-side math script or CDN.
hexo.extend.generator.register('katex-assets', function () {
  const dist = path.join(path.dirname(require.resolve('katex/package.json')), 'dist');
  const files = ['katex.min.css', ...fs.readdirSync(path.join(dist, 'fonts'))
    .map(file => `fonts/${file}`)];

  return files.map(file => ({
    path: `libs/katex/${file}`,
    data: () => fs.createReadStream(path.join(dist, file))
  }));
});
