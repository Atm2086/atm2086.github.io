---
title: learn xmake
top: false
cover: false
toc: true
mathjax: true
date: 2025-12-23 08:51:52
password:
summary:
tags: 构建系统 xmake
categories: Build System
---

起因： InifitCore 和 InifitLM 都是基于Xmake构建的，Xmake 基于 Lua 的轻量级跨平台构建工具，使用 xmake.lua 维护项目构建，相比 makefile/CMakeLists.txt，配置语法更加简洁直观，对新手非常友好，短时间内就能快速入门，能够让用户把更多的精力集中在实际的项目开发上。

功能上：既能够像 Make/Ninja 那样可以直接编译项目，也可以像 CMake/Meson 那样生成工程文件，还有内置的包管理系统来帮助用户解决 C/C++ 依赖库的集成使用问题。（直接编译项目： 将长串编译命令变成使用xmake的短行； 工程文件： 看作不同软件管理代码的账本，xmake可以生成对应的； 内置包管理： 在.lua的配置文件里写一行后会自动配置）

# 快速上手部分，剩下的得去主动学文档

https://xmake.io/zh/guide/quick-start.html#windows

## 安装与更新

最好不要在root权限（最高管理权限）下安装，减少root下误操作系统文件的风险。

更新
```bash
xmake update 2.7.1
```

1. Linux下安装

```bash
sudo apt install xmake
```

2. 源码编译安装

Git Submodule 和 Github 打包机制：

- Git Submodule 就像是代码里的“快捷方式”：

    - 主项目（xmake）的文件夹里，其实并没有存那些第三方库的真实代码。

    - 它只存了一个链接，**指向另一个存放代码的仓库**，并记录了“我需要那个仓库的哪个版本”。实现单独维护，解耦。

- 在 GitHub 页面上点击绿色的 "Code" 按钮，然后选 "Download ZIP"，GitHub 的系统非常“笨”：

    - 只打包当前仓库的文件，对待子模块等于只看名字，导致项目里很多文件夹是空的。

    - 实际是静态快照，只打包当前分支。

```bash
git clone --recursive https://github.com/xmake-io/xmake.git
cd ./xmake

./configure # 环境检查，会生成 Makefile 文件记录这台电脑的编译规则（对应后续的make指令）
make -j4    # j代表job，j4代表同时开启4个CPU核心进行并行编译
./scripts/get.sh __local__ __install_only__  # 执行它自己提供的脚本 参数是__local__ 指当前用户目录， __install_only__ 只管把编译好的文件拷贝过去
source ~/.xmake/profile  # source读取并运行指定文件里的配置， profile里大抵是记录了路径信息
```

## 使用

1. 创建工程

```bash
xmake create hello
```

创建了一个具有简单工程结构的工程：

```
hello
├── src
│   └─main.cpp
└── xmake.lua  # 工程描述文件，制定规则等等
```

2. 构建工程

```bash
cd hello
xmake
```

这里注意，如果出现 `error: /home/usr/miniconda3/envs/myenv1/bin/x86_64-conda-linux-gnu-ld: unrecognised emulation mode: 64 Supported emulations: elf_x86_64 elf32_x86_64 elf_i386 elf_iamcu` ld是Linker链接器，这里报错显示来自miniconda，conda环境为了保证跨平台的一致性，自带了一套编译器工具链比如gcc，ld

Xmake 默认会尝试使用系统最快的工具。但在 Conda 环境激活状态下，系统路径被 Conda 劫持了。xmake 找到了 Conda 提供的链接器，但这个链接器可能在参数传递上与你系统自带的编译器不匹配，导致它“不认识 64 位模式”。

劫持：当你执行 conda activate myenv1 时，Conda 修改了环境变量 PATH。它把 .../envs/myenv1/bin 这个目录插到了最前面。导致输入 g++ 或者 ld 时候会从conda的目录找。

- 退出conda模式 `xmake clean -a` 清缓存重新xmake

- 如果有必要在conda环境下工作，可以传递参数

```bash
xmake f -p linux -a x86_64 --sdk=/usr # f是config缩写，--sdk=/usr强制去系统标准目录找工具，这里指定xmake标准
xmake
```

- 安装conda的完整开发包

```bash
conda install gxx_linux-64
```

3. 运行程序

```bash
xmake run
```

4. 调试程序

xmake支持多种调试器，lldb、gdb、windbg等等

```bash
xmake f -m debug # xmake f --debugger=gdb 可以指定调试器
xmake

xmake run -d hello
```