# 仓库整理记录

核查日期：2026-09-16。以下分支关系来自成功执行 `git fetch origin --prune` 后的远程引用，并用 GitHub 公开 API 和 `git ls-remote --symref origin HEAD` 核对了默认分支。

## 整理前的分支快照

本地只有一个开发分支 `hexo`，其他名称是本地缓存的远程分支。`hexo` 是 GitHub 默认分支，也是当前源码主线；不是看到 `main` 就应切过去开发。

| 远程分支 | 提交 | 与 `hexo` 的差异 | 建议 |
| --- | --- | --- | --- |
| `hexo` | `6d8dbaa`，2026-01-17 | 基准；包含最新文章和主题定制 | 保留为唯一长期源码分支 |
| `master` | `6d8dbaa`，2026-01-17 | 0 个独有提交，内容完全相同 | 可以直接删除，无需合并 |
| `main` | `7e44069`，2025-10-28 | 独立历史，只有生成网页；提交说明指向旧源码 `8574ef8` | 归档后可清理，不要合并到源码 |
| `dependabot/npm_and_yarn/hexo-8.1.2` | `9da4fd6`，2026-05-07 | 基于最新源码，新增 1 个依赖升级提交 | 暂不直接合并；先升级构建用 Node 并验证 |
| `dependabot/npm_and_yarn/hexo-renderer-marked-7.0.1` | `61b832c`，2025-11-12 | 缺少最新 1 个源码提交，新增 1 个依赖升级提交 | 可作为独立升级候选，需验证文章渲染 |

两个 Dependabot 分支都只修改 `package.json` 和 `package-lock.json`，没有独有文章或主题定制。关闭它们不会丢失写作成果，但会放弃尚未验证的升级；“无需保留长期分支”不等于“依赖永远不必升级”。

对应 PR：[Hexo 8，#5](https://github.com/Atm2086/atm2086.github.io/pull/5)、[Markdown 渲染器 7，#1](https://github.com/Atm2086/atm2086.github.io/pull/1)，目标分支均为 `hexo`。

## 部署与分支关系

仓库中的 `.github/workflows/deploy.yml` 在 `hexo` 有新推送时构建 `public/`，通过 `actions/upload-pages-artifact` 和 `actions/deploy-pages` 发布，不需要一个专门保存 HTML 的分支。

整理前，GitHub 公开部署记录显示，最近一次 `github-pages` 部署创建于 2026-01-17，来源是 `hexo` 的 `6d8dbaa61daeed46d4b21e53030dcefe9da5ba78`，状态为 `success`：

- [部署记录 API](https://api.github.com/repos/Atm2086/atm2086.github.io/deployments?per_page=3)
- [仓库工作流](https://github.com/Atm2086/atm2086.github.io/actions)
- [整理前成功的 Pages 发布](https://github.com/Atm2086/atm2086.github.io/actions/runs/21086649741)

公开 `/pages` API 返回 404，未能直接读取后台设置。2026-09-16，用户已在 Settings → Pages 确认发布来源为 GitHub Actions；整理提交 `4f81b26` 的 [Pages 构建与部署也已成功](https://github.com/Atm2086/atm2086.github.io/actions/runs/35048462705)。据此完成旧发布分支清理。

整理前 `_config.yml` 的 `deploy.branch: main` 和 `npm run deploy` 属于另一套旧的 Git 推送部署入口。这正是 `main` 与源码主线混淆的来源之一。本轮本地修改已将 `deploy` 清空并移除该 npm 命令；GitHub Actions 工作流不受影响。

## 已确认的整理方案

用户已确认：保留 `hexo`，提交推送整理内容，删除重复的 `master`，在确认 Pages 设置后删除旧 `main`。两个 Dependabot PR 暂时保留，不合并、不关闭。

1. 保存所有分支的 Git bundle 备份，验证可恢复。
2. 保留 `hexo` 名称可以直接沿用默认分支和现有部署配置，减少迁移步骤。
3. 移除旧 Git 部署命令及 `_config.yml` 中的旧部署目标，再提交并通过一次 Pages 部署验证。
4. 删除与 `hexo` 完全重复的远程 `master`。
5. 确认 Pages 使用 Actions 后，删除已备份的旧网页分支 `main`；不用合并独立历史。
6. 保留两个依赖 PR，后续单独验证升级。Hexo 8 的锁文件声明 Node `>=20.19.0`，现有 Node 18 工作流不满足要求；渲染器升级会把 `marked` 从 4.3.0 升到 15.0.12，需要检查代码块、图片、目录和 HTML 渲染。
7. 本地已将 Dependabot 从每天、最多 20 个版本升级 PR 调整为每周、最多 3 个；推送后生效，不会自动关闭已有 PR。

本次不迁移默认分支名称，继续使用 `hexo`。

## 备份与恢复

整理前所有已获取分支的备份保存在本机 `.local-backups/before-branch-cleanup-2026-09-16.bundle`，该目录已忽略，不会发布进博客。备份包含已提交的 Git 历史和引用，不包含未提交文件；原有未跟踪 `.mcp.json` 保留原位。

检查备份：

```bash
git bundle verify .local-backups/before-branch-cleanup-2026-09-16.bundle
git bundle list-heads .local-backups/before-branch-cleanup-2026-09-16.bundle
```

例如，日后需要从备份恢复旧网页历史到本地独立分支时：

```bash
git fetch .local-backups/before-branch-cleanup-2026-09-16.bundle refs/remotes/origin/main:refs/heads/recovered-old-pages
```

恢复命令不改动当前工作区，也不推送 GitHub。该备份只在这台机器上，可另行复制到自己的备份介质。

## 执行结果（2026-09-16）

已完成：更新远程引用，核对分支历史和源码目录，检查公开默认分支、PR、成功部署记录，新增目录与日常开发说明，忽略本机工具配置和备份目录，保存并验证整理前备份，禁用本地旧部署入口，降低 Dependabot 检查频率和版本升级 PR 上限。

整理提交 `4f81b26` 已推送到 `hexo`，Pages 构建与部署成功。远程 `master` 和旧 `main` 已删除，删除时均校验分支仍指向备份中的提交，并已通过 `git fetch origin --prune` 清理本地远程引用。

最终保留的远程分支：

- `hexo`：唯一长期开发分支，仍是默认分支和 Pages 发布来源。
- `dependabot/npm_and_yarn/hexo-8.1.2`：保留，对应 PR #5。
- `dependabot/npm_and_yarn/hexo-renderer-marked-7.0.1`：保留，对应 PR #1。

两个依赖 PR 均未合并或关闭。上表是整理前快照，便于追溯和恢复；日常写作直接使用 `hexo`。
