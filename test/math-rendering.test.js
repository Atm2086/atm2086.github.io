'use strict';

const { before, after, test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Hexo = require('hexo');

const hexo = new Hexo(path.resolve(__dirname, '..'), { silent: true });
before(() => hexo.init());
after(() => hexo.exit());

function render(text) {
  return hexo.render.renderSync({ text, engine: 'md' });
}

test('inline TeX survives Markdown escapes and Chinese punctuation', () => {
  const html = render(String.raw`中文$x_0 = \bot$，以及 $ f'(x) = \frac{a_1}{b_2} $。`);
  assert.equal((html.match(/class="katex"/g) || []).length, 2);
  assert(html.includes(String.raw`x_0 = \bot</annotation>`));
  assert(html.includes(String.raw`f&#x27;(x) = \frac{a_1}{b_2}</annotation>`));
  assert(!html.includes('katex-error'));
});

test('display math works immediately after prose and inside lists', () => {
  const html = render('抽象域：\n$$\nL = \\left\\{ \\bot, Z^{-0}, \\top \\right\\}\n$$\n\n- 推导\n\n  $$\n  x_0 \\leq x_1\n  $$');
  assert.equal((html.match(/class="katex-display"/g) || []).length, 2);
  assert(!html.includes('katex-error'));
  assert(html.includes(String.raw`\left\{`));
  assert(!html.includes('$$'));
});

test('code, escaped dollars and Mermaid remain literal', () => {
  const html = render('`$x_0$` and \\$5\n\n```bash\necho "$HOME"\n```\n\n```mermaid\nflowchart TD\n  A --> B\n```');
  assert(!html.includes('class="katex"'));
  assert(html.includes('<code>$x_0$</code>'));
  assert(html.includes('$HOME'));
  assert(html.includes('$5'));
  assert(html.includes('class="mermaid"'));
});

test('existing lattice and matrix articles render all their math', () => {
  for (const relative of [
    'Compiler/格理论和单调框架.md',
    'DL-Framework Learning & infra learning/Framework自制2.md'
  ]) {
    const source = fs.readFileSync(path.join(hexo.source_dir, '_posts', relative), 'utf8');
    const body = source.replace(/^---\n[\s\S]*?\n---\n/, '');
    const html = render(body);
    assert(!html.includes('katex-error'), relative);
    assert(!html.includes('$$'), relative);
    assert(html.includes('class="katex-display"'), relative);
    if (relative.includes('Framework')) {
      assert(html.includes(String.raw`\begin{bmatrix}`));
      assert.equal((html.match(/class="katex"/g) || []).length, 9);
    }
  }
});

test('formula styles and referenced fonts are generated locally', async () => {
  const routes = await hexo.extend.generator.get('katex-assets').call(hexo);
  const files = new Set(routes.map(route => route.path));
  assert(files.has('libs/katex/katex.min.css'));
  const dist = path.join(path.dirname(require.resolve('katex/package.json')), 'dist');
  const css = fs.readFileSync(path.join(dist, 'katex.min.css'), 'utf8');
  for (const [, font] of css.matchAll(/url\((fonts\/[^)]+)\)/g)) {
    assert(files.has(`libs/katex/${font}`), font);
  }
});
