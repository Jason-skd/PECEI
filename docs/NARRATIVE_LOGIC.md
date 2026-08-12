# AIGI - 

## Introduction

背景: 世界模型, AI 孪生, 具身智能, fine-tunning, LoRA
人类在为各行各业训练专精于某个领域的人工智能
而我们组认为, 我们只需要一个通用型人工智能, 它自己就能分化 (胚胎干细胞) 产生针对特定任务的人工智能 —— Artificial Intelligence Generated Intelligence 范式
我们组重点以最特殊的 AI, 具有物理实体意义的 —— emboddied intelligence 为研究目标, 探索一种通用型人工智能能否通过自我 generate/regenerate 产生针对于各种场景的专精智能?
(补充这方面的相关学术研究)

## Methodology

a game of **real world**

我们专门精心设计了一个 2D 网格世界 + LLM 无梯度学习方案
回答两个问题:

### 为什么这个 2D-Grid world 能够替代我们在物理世界做真实研究? 因为它足够真实: (后期附上游戏构造流程图)

1. 2D-Grid 实际上是一个关注智能如何指挥硬件的高效抽象: script 的原语: act() 和 beat() 是真实 emboddied intelligence 的 intelligence 方案, 它并不负责调控身上每一处机械如何运作, 这一块工程细节也恰好被我们的网格世界给抽象化省去了
2. 环境足够真实: 一个真实的救援场景, 一场地震/辐射/山火灾难中, emboddied intelligence 如何完成救援任务?

### 我们的 AI 如何学习的? (后期附上 prompt enginering 图)

1. 核心架构: 三重循环 (见 ROADMAP)
2. buffer 化上下文压缩 score 化的 complexity 引导的经验管理: 没有 snowball
3. RPG 做长期记忆存储与取回: 自进化范式 (暂时没做)

## Results

我们的原型机具有学习效力:

几张图, 图一: 对比实验, 经历了其他地图学习的原型机, 相较于直接来探索复杂地图的原型机, epoch 消耗大幅减少, 花费的 round 也更少 (更快熟悉地图、救出人类), 体现了学习的进步
图二: 针对最复杂场景, 最终给出的 script, 发现 LLM 充分利用了我们脚本设置的所有关键词, 写出了复杂的 if-else, for, while, observe-reaction 语句, 是图灵完备的, 证明了我们的原型机真的能在特殊场景下特化
图三: 补充实验: 上一张地图生产的 script 放到下一张 map 实际上无法跑通, 证明这是真的特化的 mind, 而非又一个 general mind

## Concludes: 最终, 我们的这套学习方案, 足以被称为「原型机」(prototype)

**项目真正贡献, 真正亮点**, 因为这套范式:
1. 最重要, 后端无关, 我们用的是最简单的模拟: LLM API, 但是换智能后端, 比方说 RL, AGI, 这套学习范式都能够支撑 embodied intelligence 针对性特化
2. 我们的 prototype 有自 generate (暂时没做) 和 regenerate 的能力 (体现在 script 修正, 对应 regenerate mind)
3. 我们的 prototype 真的在特化自己, 特化的同时又没有丢失自己的 “再分化” 能力, 反而是特化的过程成就了它的学习

## Discussion:

几个不足:

1. 2D-Grid 足够抽象, 但是缺少细节, 也不能说明这层抽象挪用到 embodied intelligence 身上一定就成功. 然而这是我们夏令营 13 天在受限的硬件和算力支持下, 抓住核心问题的权衡方案
2. 我们用消耗 epoch, Round 来体现原型机真的在学习, 并没有真实的 benchmark —— 学术界目前确实也没有, 据我所知, 欢迎教授们指点
3. 我们宣称的「后端无关」尚未经过验证: 我们只用了 LLM API 这一种后端, 并没有换上 RL (强化学习)、监督学习 (Supervised Learning) 等其它方法做对照实验, 来证明这套学习范式确实与后端无关. 受限于夏令营 13 天的时间与有限的算力, 我们无法完成这样的对照试验

我们这套原型机令人激动的野心 (未来展望):

1. 数字居民
2. AGEI —— 通用型具身智能
3. 终极图景: AIGI —— 真的有一天 AI 制造出了另一个 AI, AI 的繁衍
(阐明一下在我们的 prototype 之上补充上什么工作就可能实现)

## Acknowledege and reference

...