# Fight for the future

个人静态博客，使用 Hexo 7 和仓库内的 Matery 主题。

- 网站：<https://atm2086.github.io>
- 当前源码主线和 GitHub 默认分支：`hexo`
- 发布入口：向 `hexo` 推送提交，由 GitHub Actions 构建并部署 Pages。
- 分支现状、清理依据和恢复方式：[仓库整理记录](docs/repository-guide.md)。

## 目录导航

| 路径 | 用途 |
| --- | --- |
| `source/_posts/*.md` | 博客文章，目前 18 篇 |
| `source/_posts/文章名/` | 对应文章的图片等附件，应与文章一起保留 |
| `source/about/`、`categories/`、`tags/`、`friends/`、`contact/` | 独立页面，均位于 `source/` 下 |
| `source/_data/friends.json` | 友情链接数据 |
| `_config.yml` | 站点标题、URL、文章链接、分页、主题等配置 |
| `themes/hexo-theme-matery/_config.yml` | 主题外观和功能配置 |
| `themes/hexo-theme-matery/layout/`、`source/`、`scripts/` | 主题模板、静态资源、扩展脚本，均位于主题目录下 |
| `scaffolds/` | 新文章、新页面的模板 |
| `package.json`、`package-lock.json` | 依赖和锁定版本，两者一起维护 |
| `.github/workflows/deploy.yml` | 自动构建和发布 |
| `.github/dependabot.yml` | 自动依赖升级 PR 配置 |
| `public/`、`node_modules/`、`db.json`、`.deploy*/` | 生成物、依赖和缓存，已忽略，无需提交 |
| `.mcp.json` | 本机工具配置，已忽略 |

`_config.landscape.yml` 是空的旧主题配置，`hexo-theme-landscape` 也是保留的旧主题依赖；当前实际使用 `hexo-theme-matery`。主题目录包含现有定制，不要直接用新版主题覆盖。

## 本地写作

安装 Node.js 和 npm 后，在仓库根目录执行：

```bash
npm ci
npm run server
```

预览地址通常为 <http://localhost:4000>。新建文章：

```bash
npx hexo new "文章标题"
```

编辑 `source/_posts/文章标题.md`，图片放在同名目录中。提交前构建：

```bash
npm run clean
npm run build
git status
```

## 单主线开发

目前直接在 `hexo` 写文章、调整主题即可，不需要把内容在 `hexo`、`master`、`main` 之间来回合并。开始前确认没有未提交修改，再同步：

```bash
git switch hexo
git pull --ff-only origin hexo
```

完成预览和构建后，只暂存本次需要发布的文件，提交并推送 `hexo`，然后检查 GitHub Actions 的 `Pages` 工作流。依赖升级如需 PR，可使用短期分支，验证并合并后删除，不必长期保留。

旧的 `npm run deploy` 命令已移除，`_config.yml` 中的 Git 部署目标已清空，避免重新生成旧 `main` 分支。日常发布只需推送源码，由 Actions 完成。旧的 `hexo-deployer-git` 依赖暂时保留，但没有启用的部署目标。

Dependabot 已调整为每周检查，版本升级 PR 最多同时保留 3 个；该配置需要推送到默认分支后生效，不会自动关闭已有 PR，也不限制安全升级 PR 的数量。

当前工作流配置 Node 18，依赖锁定 Hexo 7.3.0。升级 Node 和 Hexo 应作为单独变更验证；不能直接合并要求 Node ≥20.19 的 Hexo 8 分支后继续使用现有工作流。
