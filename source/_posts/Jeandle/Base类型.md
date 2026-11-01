---
title: Base类型
top: false
cover: false
toc: true
mathjax: true
date: 2026-10-27 22:31:19
password:
summary:
tags:
categories:
---

# JavaType Base类型

功能：取得不依赖具体查询位置（basicblock）就能保证成立的类型约束。
例子：以一个依赖具体查询位置的为例：
```
Animal obj = ...;

use(obj);                    // 位置 A

if (obj instanceof Dog) {
    use(obj);                // 位置 B
}

use(obj);                    // 位置 C
```

同一个 obj，三处位置，能够保证的类型不同：但是不依赖位置的约束，是 Animal。但同时也可以看出，有的指令会自带这种信息，有的指令会通过输入关系（use-def）传播这些信息。
| 查询位置 | 能保证的约束 | 原因 |
|---|---|---|
| A | `Animal` | 只有对象来源提供的信息 |
| B | `Dog` | 执行到这里，必然通过了 `instanceof Dog` 检查 |
| C | `Animal` | 检查成功和失败的路径都可能到达这里 |

自带信息的以 load 为例子。传播信息的以 phi 为例子。对于 branch，不会进行控制流分析，具体走了哪条边才能知道什么，而是直接查询输入依赖汇总。所以最近似的理解是，这个功能的实现是做了一个数据流分析，而不是控制流分析。对于其中循环依赖的问题，使用了SCC强连通分量进行了处理。

抽象：
- 节点：LLVM 的 Value。
- 依赖边：某个值的类型依赖哪些输入值。
- 传播的信息：JavaType 类型约束。
- 传递规则：identity 传递输入类型，PHI/select 合并输入类型，再结合节点自己的类型信息。
- SCC：把相互依赖的一组值作为整体求解，在内部持续传播，直到结果不再变化。

## 来历（seed）

1. 值定义 和 输入推导
    - 值自身携带：
        - Argument 参数属性
        - Call/Invoke 调用返回属性
        - load metadata
        - oop-handle VM 信息
    - 输入推导
        - phi/select
        - identity（恒等） 表达式转发输入
        - literal null/poison arm 在 非空oop下跳过
            - arm指的是某个选择分支
            - 字面量空指针或者毒性值跳过
            - 非空oop是指，当前分析器已经知道这个值一定不是null
        - undef 保持吸收性（freeze指令）
            - 将 undef 或者 prison 值转换成一个任意且合法的具体值
            - 需要证明 defined 才能用
        - select condition 在condition已知的情况下只选择对应的arm
2. 实现
    1. 拿到节点独立信息：`getBaseJavaTypeLeaf()`
    2. 分析类型依赖哪些输入：`getBaseTypeInputs()`
    3. 算法主体：`getBaseJavaType()`
    4. 遇到环就按 SCC 处理：`

## SCC 算法思路

无环：直接查，按路径推导

有环：将循环依赖识别为一个 SCC，{p, q}
1. 拿到覆盖他们类型约束的外环输入 a, b
2. 加上各成员自己的 seed 信息
3. 迭代到不动点

```
p = phi(a, q)
q = phi(b, p)
```