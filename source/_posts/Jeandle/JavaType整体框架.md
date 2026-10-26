---
title: JavaType整体框架理解
top: false
cover: false
toc: true
mathjax: true
date: 2026-10-26 23:55:33
password:
summary:
tags:
categories:
---

# 启发

以前理解项目的能力一直没跟上，也十分缺少方法论。

一个大模块可以拆成几个子模块，子模块又可以继续划分。当一个模块划分到合适大小时候，就可以使用ai工具统计相关函数接口和关键数据结构辅助理解。

同时设计往往强绑定需求，而需求又和功能正相关，因此从需求理解功能，从需求理解设计往往可能会更轻松。以下尝试在三小时内理解 JavaType 的整体设计，并整理出第二版 cache 实现，大规模改动有点难受，每次都让我难以下手。而代码文本本身是一个线性的过程，人的理解从个人角度来看是图结构的，所以只能说不太匹配了。不过anyway，天地一指万物一马，方法不重要，重要的是最终的理解是不是到了。

以上说的是我的方法，如有启发诚惶诚恐。

**以 JavaType 整体架构为例，结合ai，将其分为以下六个模块**：

- 类型表示和相关运算：提供共同的结果表示，大厦的数据根基。
    - `JavaType` 关键数据结构
    - `typeUnion()` `typeIntersect()`
- 对外查询接口：组织一次查询
    - `JavaTypeQueryContext` `IsNullEdgeOracle` 关键数据结构，查询上下文
    - `getJavaType()` `sharpen()`
- 约束模块
    - Base类型求解（新的SCC算法）：**只根据值本身和输入关系能确定的约束**
        - BaseMemo、BaseTypeNode等等
        - getBaseJavaTypeLeaf、getBaseJavaType
    - 条件和边约束：**通过 CFG 图的路径确定的约束**
        - TraceResult
        - traceToCheckInstanceof、edgeGuard
    - 约束组合
        - valueDimension、phiValueType、factsAt
- 一些功能模块
    - IR识别和 definedness证明

## JavaType 类型标识模块

关注点：基本类型和类型运算。

定义了 JavaType 类型，而不是一个简单的某个class这种，因为需要的是一些绑定关系，从而方便计算和运行时的真实情况。所以理解为一组约束，只有其中的 Klass 才是真正的需要的类型信息。但需要注意的是**JavaType描述对象的前提是对象非空时会是什么类型，但对象非空这件事不知道**。

| 字段 | 含义 |
|---|---|
| `Klass` | 已知的类约束；`0` 表示没有这项信息 |
| `Exact` | 是否恰好是这个类，而非其子类 |
| `Interfaces` | 已知满足的接口约束 |
| `ExcludedKlasses` | 已知不属于哪些类及其子类 |

两个基本运算：Union 和 Intersect。
- Union：值可能来自 A 或 B，保留两种可能下都成立的信息，例如 PHI 汇合。**相当于可能性增加**
- Intersect：A 和 B 同时约束同一个值，组合信息，尽可能收窄类型。**相当于可能性缩小**

两类运算具体怎么干的，暂不细究。

## 对外接口，组织一次查询

注意：这两个工具功能不同，但同样依赖一致的底层实现，但由于所调用函数的差距导致，这两个函数的实现略有不同，但功能是好确定的，因为他俩的区别是约束不一样。

- getJavaType：综合基础类型和上下文约束的结果，这个值在指定位置是什么类型。
- sharpen：仅返回路径检查产生的类型约束，不包含基础类型。也就是说，走过某个特定边的时候能额外返回的信息。
    - RecoverTypeInfo 会使用这个函数

下面代码注意：ContextBB 理解为前驱块的代码关系，edgeGuard 这个函数都会用到，typeIntersect 隐式包含了 normalizeExcludedKlasses。

```简化后的代码关系
// getJavaType：综合类型
T = valueDimension(V, Q);
F = factsAt(V, ContextBB, Q);
return typeIntersect(T, F);

// sharpen：指定入边上的路径约束
F = factsAt(V, ContextBB, Q);
mergePathFacts(F, edgeGuard(ContextBB, DestBB, V, Q));
normalizeExcludedKlasses(F);
return F;
```

## 约束

### 基本约束，Base求解，沿SSA依赖计算基础类型，SCC

具体思路：先读取直接信息，再处理依赖。主要接口：`getBaseJavaTypeLeaf` 和 `getBaseJavaType`.
- `getBaseJavaTypeLeaf(V)` 读取值自身携带的信息
- `getBaseJavaType(V, BaseMemo, DefinedMemo)` 组织依赖求解，并处理循环依赖
    - basememo 提供复用
    - 循环依赖处理：SCC工作处（**待办，有一些复杂的结构**）

```
参数属性、调用返回属性、load metadata 等
                   ↓
             直接类型信息
                   ↓
PHI / select / identity / freeze 的输入关系
                   ↓
              Base 类型结果
```

### 条件约束，控制流（仍是递归）

需要根据控制流路径条件翻译出约束。

标志性的接口：
- `edgeGuard(From, To, V, Q)`，根据**具体走哪条边**，返回该边对 V 的约束。
- `traceToCheckInstanceof`，读懂条件（因为可能走不同边），算出不同结果各自意味着什么，针对条件表达式。
- `TraceResult` 结构体，用来记录一个条件表达式在不同结果下，对对象产生的类型约束。

### 约束组合，相当于不同的组合模式

- `factsAt(Value *V, BasicBlock *BB, Q)`，到达基本块BB时候，路径检查对V的约束（只有 CFG 约束）
- `valueDimension(Value *V, Q)`，Base + 输入来源处的约束
    - `phiValueType(PHINode *PN, Q)`，合并各incoming值在对应边上的类型，它是 valueDimension 处理 PHI 的子步骤

两者通过一次查询共用的 JavaTypeQueryContext Q 联动：

| Q 中的成员 | 数据形态 | 用途 |
|---|---|---|
| `DT`、`NullOracle` | 支配树引用、回调 | 判断支配关系、证明空值边 |
| `BaseTypes` | `Value* → JavaType` | Base 结果 |
| `DefinedValues` | `Value* → bool` | definedness 证明 |
| `ValueTypes` | `Value* → JavaType` | 可在本次查询内复用的值维度结果 |
| `Processed` | `(Value*, BB*) → optional<JavaType>` | 路径结果，同时检测正在计算的递归节点 |
| `PhiVisited`、`ValueVisited` | 指针集合 | 当前递归栈状态 |
| `FactsBudget`、`PhiBudget`、`ConditionDepth` | 计数器 | 控制路径、值维度和条件追踪的计算范围 |