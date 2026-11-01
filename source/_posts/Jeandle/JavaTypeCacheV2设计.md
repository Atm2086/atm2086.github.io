---
title: JavaTypeCacheV2设计
top: false
cover: false
toc: true
mathjax: true
date: 2026-10-30 09:49:12
password:
summary:
tags:
  - Jeandle
  - JavaType
  - LLVM
  - 缓存设计
  - 缓存失效
categories:
  - Jeandle
---

# 总设计

目前暂时默认JavaType的缓存没有内存压力，因此暂时没做那种带限制 densemap 个数的。

设计：独立的不同类型的memo cache，然后不同cache下放针对各个memo的测试工具和 invalidate 方法，以及其他的涉及到算法的相关依赖，最后总的 JavaTypeCacheV2 作为 Analysis 工厂方法产物交给 FAM 保管。
```
JavaTypeCacheV2：

    - BaseCacheV2
        - DenseMap<Value *, JavaType>

    - DefinedCacheV2
        - DenseMap<Value *, bool>

    - FactsCacheV2
        - std::pair<Value *, std::pair<BasicBlock *, BasicBlock *>>
        - DenseMap<Key, JavaType> Guards
        - DenseMap<Key, bool> NullEdges

    - FinalCacheV2
        - using Key = std::pair<Value *, BasicBlock *>
        - DenseMap<Key, Entry> Memo;

```

## 总的失效算法设计 污点传播

主要思想：先拿到修改点，然后把修改点扩展为可能受到影响的所有值。
- 如果输入改了，要沿着 users 找谁依赖他
- Value 作为 LLVM IR 基类的多态：
    - 修改或新生成的指令 Instructions
    - 被删除或者被替换的变量 Values
    - 被删除或者被替换的块 Blocks
    - 函数的参数 Arguments

所以：
1. 通过参数拿到修改点
2. 利用 LLVM 的 use-def 链条提取受影响范围
3. 把提取的东西喂给四层缓存，让他们分别根据自己的invalidate来失效

- Pending 待处理队列
- Affected 确认为受影响的值
- Blocks 受影响的基本块列表
- SeenBlocks 已访问的基本块集合
- Seen 已访问的值集合

```
JavaTypeCacheV2::invalidate(ArrayRef<Value *> Changed, bool CFGChanged,
                            JavaTypeBaseInvalidationMode Mode){
    // 1. 初始化核心容器：Seen 用来去重，Affected 和 Blocks 用来收集最终的失效范围
    SmallPtrSet<Value *, 32> Seen;
    SmallPtrSet<BasicBlock *, 32> SeenBlocks;
    SmallVector<Value *, 32> Affected;
    SmallVector<BasicBlock *, 16> Blocks;

    // 工作队列：把外面传进来的改变点（震中）作为算法的初始源头
    SmallVector<Value *, 32> Pending(Changed.begin(), Changed.end());

    // 2. 核心图遍历循环（污点扩散）
    while (!Pending.empty()) {
        Value *V = Pending.pop_back_val();      
        // 去重防护：如果这个值已经处理过，或者为空，直接跳过（防止死循环）
        if (!V || !Seen.insert(V).second)
            continue;
        // 情况 A：如果是基本块（控制流）变了
        if (auto *BB = dyn_cast<BasicBlock>(V)) {
            if (SeenBlocks.insert(BB).second)
            Blocks.push_back(BB); // 收集受影响的块          
            // 块变了，里面的指令全部连带失效
            for (Instruction &I : *BB)
                Pending.push_back(&I);
            continue;
        }
        // 情况B：指令变了 或者 参数变了
        if (auto *I = dyn_cast<Instruction>(V)) {
            // 
            AddBlock(I->getParent()); 
            Affected.push_back(I);
        } else if (auto *A = dyn_cast<Argument>(V)) {
            Affected.push_back(A);
        } else {
            // 无法识别的其他类型，触发 FullClear（此处省略）
            continue;
        }
        for (User *U : V->users()) {
            Pending.push_back(U);
        }
    }

    // 3. 最终应用：用提取出的精细化范围，让不同层级的缓存精准失效
    Base.invalidate(Affected, CFGChanged, Mode);
    Defined.invalidate(Affected, CFGChanged, Mode);
    Facts.invalidate();
    Final.invalidate(Affected, Blocks, CFGChanged);
}
```

CFGChanged：用来删除所有非叶子项。
Affected：触及到的应该修改的项。

## BaseMemo

参数：
| 参数 | 用途 |
|---|---|
| `Affected` | 外层已经展开好的受影响 Value 列表，Base 不再遍历 users |
| `CFGChanged` | 是否发生 CFG 结构变化，决定是否额外删除非叶子缓存 |
| `Mode` | 选择**如何找到要删除的项** |
| `SlotsScanned` | 输出统计：本次遍历检查了多少个已有缓存项，不参与失效判断 |

```
// 入口
size_t JavaTypeBaseCacheV2::invalidate(ArrayRef<Value *> Affected,
                                     bool CFGChanged,
                                     JavaTypeBaseInvalidationMode Mode,
                                     size_t &SlotsScanned)

// 具体实现，模板写法
template <typename MapT, typename LeafPredicate>
size_t invalidateValues(MapT &Memo, ArrayRef<Value *> Affected,
                        bool CFGChanged, JavaTypeBaseInvalidationMode Mode,
                        size_t &SlotsScanned, LeafPredicate IsLeaf)
```

Base 的 叶子 和 非叶子定义：叶子指 Base / Defined 求解算法里直接得出结果、不继续依赖其他 Value 推导的节点，比如下面的 a，其余都是非叶子。
```
%a = load ptr, ptr %addr
%b = bitcast ptr %a to ptr
%c = phi ptr [ %b, %bb1 ], [ %other, %bb2 ]
```

修改模式：
- 默认：直接删
    - 缓存很大，受影响很小时候更快
- 可选：顺序判断是否为受影响，再删
    - 受影响非常多几乎和缓存一样时快


## DefinedMemo

和 BaseMemo 保持一致。

## FactsMemo

目前全清

## FinalMemo

