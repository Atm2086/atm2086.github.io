---
title: Value
top: false
cover: false
toc: true
mathjax: true
date: 2026-10-31 09:43:27
password:
summary:
tags:
  - LLVM
  - LLVM IR
  - Value
  - Use-Def
categories:
  - LLVM架构设计
---

# Value

LLVM IR 中万物皆 Value，只有 Value 能被其他指令当作 Operand 来使用。而一旦被别人使用，使用它的就能通过 `V->users()` 来找失效范围。

```
define i32 @my_function(i32 %arg0) {    ; 1. %arg0 是一个 Argument（它是 Value）
entry:                                  ; 2. entry 是一个 BasicBlock（它是 Value）
  %t1 = add i32 %arg0, 10               ; 3. %t1 是一个 Instruction（它是 Value）
  store i32 %t1, i32* %ptr
  ret i32 %t1
}
```

代码里通常出现进行转换。
```
if (auto *BB = dyn_cast<BasicBlock>(V)) { ... }
else if (auto *I = dyn_cast<Instruction>(V)) { ... }
else if (auto *A = dyn_cast<Argument>(V)) { ... }

```