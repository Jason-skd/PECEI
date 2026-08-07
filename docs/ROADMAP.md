# 项目路线图：Embodied Intelligence — Generate / Regenerate

> 14 天剑桥夏令营 · AI/CS 方向 · 交付物：Poster + Presentation（无需现场 live demo）

---

## 0. 一句话定位

我们在一个**网格世界后端**上,构建一个具身智能体,它**生成并再生(generate / regenerate)** 自己的造物——而"造物"同时包含**躯体(body)**与**心智(mind)**。世界被刻意设计成"智能体本体够不到目标"的形态,迫使它必须**造出一个助手、为它写大脑、再与它协作**;当环境打破造物的假设时,智能体**重写躯体或心智,再试**。

由于缺乏真实机器人硬件与 world model 的算力,网格世界是我们对真实具身场景的**有原则的等效替代(principled surrogate)**;其抽象层与真实机器人的 primitive 层同构,因此整套 generate/regenerate 架构是一个**可迁移的 prototype**。

---

## 1. 核心命题 (Thesis)

> **Mind and Body, Both Authored.**
> 一个具身智能体不仅生成、而且要**再生**它的造物;造物同时包含**躯体**(由材料合成的物理形态)与**心智**(智能体为造物编写的控制程序)。环境既**物理地强制**智能体依赖造物(本体进不去),又**动态地强制**它在失败后重造/重写。

**两个动词的精确定义:**

- **Generate**:在某个决策点,智能体为造物合成两份 artifact——
  - **Body**:一份**类型化的材料规格**(材料组合 + 导出属性 + 物理约束),随后在场景中由材料拼装而成。
  - **Mind**:一段**类型化的控制程序**(DSL),驱动该造物行动。
- **Regenerate**:当任一 artifact 失效时,智能体修复它。失效分两层(见 §4):**编译期**(静态检查未过)或**运行期**(在世界中执行失败)。借助**故障定位(fault localization)**,它只重写出错的那一处,而非整体推倒重来。跨 episode 累积的反思与可复用技能,使每一次 regenerate 都比上一次更有依据。

---

## 2. 方法论总纲:网格世界为何"等同"于真实具身实验

这是整个方案必须先立住的逻辑,也是 mentor 最可能追问的地方。

**核心论点:** 智能体的"心智"(策略程序)与"躯体规格"被表达成一种**后端无关(binding-agnostic)**的形式。网格世界只是"世界模型 + 物理"的一个 **mock 后端**。把后端从 gridworld 换成真实机器人的 primitive API,**整个 generate/regenerate 控制结构与记忆机制原样保留**。因此我们的贡献(架构 + 学习机制)是硬件无关的;网格世界只是"买不起机器、也跑不起 world model"时,仍能完整跑通这条闭环的**最廉价后端**。

**抽象层同构表(辩护词):**

| 网格世界抽象                   | 真实具身对应                         | 为何可迁移                                  |
| ------------------------------ | ------------------------------------ | ------------------------------------------- |
| 离散网格 + 单元占用            | 占用栅格 (occupancy grid)            | 本就是真实机器人导航的真实表示              |
| 动作 DSL(`move`/`turn`/`pick`) | 运动原语 (motion primitives)         | 真实控制栈即如此分层                        |
| 材料 + 属性(防火/防水/浮力)    | 物体的 affordance / 材料属性         | 同一套符号推理                              |
| 由材料拼装的造物               | 机器人**制造/部署的工具或子智能体**  | 真实 tool-use / 可部署子体                  |
| 心智 = 控制程序                | 可执行的机器人控制程序               | **1:1**,同一程序换底层 binding 即可驱动机器 |
| generate → fail → regenerate   | 造工具+写程序 → 感知失败 → 重造/重写 | 控制结构完全相同                            |

**诚实的边界(主动划清,反而更可信):**
网格世界**不**模拟连续动力学、摩擦、感知噪声、真实制造。因此本 prototype 验证的是**决策/控制层的迁移**,而非物理保真度。物理保真度正是 world model 该补的洞——**而这恰好就是我们买不起的东西**。动机与方法的缺口在此闭环。

---

## 3. 系统架构

```
┌─────────────────────────── 后端:环境(mock backend)───────────────────────────┐
│        Gymnasium API + MiniGrid 自定义救援网格世界                            │
│  agent 本体 · 被困幸存者 · 危险(火/水/窄洞) · 材料库(防火/防水/浮力)          │
│  ★ 至少一关:本体被物理挡住,不造物绝对过不了(forced-dependency level)        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ 结构化 observation(占用栅格 + 状态文本)
                                   ▼
┌──────────────────────── 前端:LLM Agent(大脑,Claude API)──────────────────────┐
│  控制模式:program-ahead + 感知 + 失败重入(非逐帧 reactive)                  │
│                                                                                │
│  决策点产出两份 artifact:                                                      │
│    ① Body = 类型化材料规格  ──►  静态 checker(编译期)                         │
│    ② Mind = 类型化 DSL 程序 ──►  静态检查通过后,沙盒解释器在世界中执行(运行期) │
│                                                                                │
│  失败处理:                                                                     │
│    编译期失败 → schema/类型约束,生成时即挡                                     │
│    运行期失败 → 故障定位 + 局部 regenerate(fault-localized regeneration)      │
│                                                                                │
│  跨 episode 记忆:Reflexion 反思库 + VOYAGER 式技能库(= verbal RL)            │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ 安全 DSL(结构化输出 / 受控 AST)
                                   ▼
                          沙盒执行器 → env.step → 渲染 / 录像 / 轨迹
```

**两层划分要点:**
- **后端(环境)**:负责世界状态、物理规则(离散)、渲染、录像。可被替换为真实机器人 API 而不影响前端。
- **前端(智能体)**:负责生成、再生、记忆。硬件无关,是本项目的真正贡献。

---

## 4. Generate / Regenerate 机制(方案核心)

### 4.1 两份生成的 artifact

| artifact       | 形态                                                               | 校验时机                                                   | 失效含义                                 |
| -------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------- |
| **Body(躯体)** | 类型化材料规格:`{材料集合, 导出属性(防火/小型/浮力...), 物理约束}` | **编译期**:静态 checker 校验材料兼容性、属性推导、约束满足 | 规格自相矛盾(如"防火但用了易燃材料")     |
| **Mind(心智)** | 类型化 DSL 控制程序                                                | **编译期**:类型/语法/schema 校验;**运行期**:在世界中执行   | 程序逻辑错(撞墙、卡死、超时、造物被烧毁) |

> 设计上让 body 与 mind **对称**:两者都是生成的 artifact,都可能是 regenerate 的对象。这使 generate/regenerate 有清晰的层次,也天然适配 PL/AI4SE 的工具链(见 §7、§11)。

### 4.2 失败分层

- **编译期失败(compile-time)**:body-spec 或程序通不过静态检查 → 在**生成阶段**就挡掉。
  - 实现手段:API 路线用 **tool-use / JSON schema 约束输出**(让 LLM 吐符合 schema 的 AST,schema 合法即可解析、基础类型合法);本地路线可进一步做 **type-directed decoding**(逐 token mask)。
- **运行期失败(runtime)**:通过了静态检查,但在世界中执行失败 → 走**回溯自修复**。

### 4.3 故障定位再生(fault-localized regeneration)

运行期失败不整体重写,而是:
1. 从**执行轨迹尾部**定位出错的程序节点 / 造物部位;
2. 只让 LLM **regenerate 那一处**(self-repair / backtracking debug 范式);
3. 重新部署,直到成功或耗尽预算。

这是 generate/regenerate 最有说服力的演示形态:**不是一锅粥地重写,而是"先编译期拦截,再运行期定位修复"**。它直接复用团队在 PL/AI4SE 方向的积累。

### 4.4 强制依赖关卡(forced-dependency level)

至少一个关卡,把"必须依赖造物"做成**硬性的关卡规则**,而非叙事口号:
- agent 本体被物理挡住(体型下限使它进不了 1 格窄洞 / 本体不防火无法穿过火区);
- 唯一解:造一个**体型更小 / 防火**的助手,为它写 mind,再与之协作把幸存者救出。

这使主题"可见、可演示、可量化"。

---

## 5. 学习机制:Verbal RL(无梯度学习)

用 LLM API 即意味着**无法做梯度学习**。"学习/积累经验"只能由 prompt/harness 层的记忆机制实现。采用两条已被验证的、且本身就属于"用语言模拟 RL"的机制:

- **Reflexion(文字反思)**:每次失败后,LLM 写一段反思(哪里错、下次怎么改),存入记忆库;后续决策时检索 top-k 相关反思塞入 prompt。跨 episode 提升,零梯度。→ 直接回答"如何让 agent 从失败中学习"。
- **VOYAGER 式技能库**:成功的 mind-程序被存档、索引、复用。"学习"= 一个不断增长的程序库。→ generate/regenerate 的元层面:生成技能 → 失败就 regenerate → 好的留下来。

**上下文窗口会不会爆——是已解决的工程问题,前提是不犯"全量历史拼接"的反模式:**

| 组成                              | 估算 tokens |
| --------------------------------- | ----------- |
| 网格 observation(占用栅格 + 状态) | ~200–300    |
| mind 程序                         | ~100–800    |
| 失败轨迹尾部(k≈20 步)             | ~1,000      |
| Reflexion 反思 top-5              | ~750        |
| 技能签名                          | ~500        |
| **单次决策 prompt 合计**          | **~2–4k**   |

相对 Haiku 4.5 的 200k 上下文有 50–100 倍余量。**只要用"检索 top-k + 反思摘要"而非全量拼接,上下文就不是技术障碍。**

---

## 6. 实验设计(产出海报证据)

三方对照,在同一后端上比较不同"学习范式":

| 条件                             | 智能体                         | 学习来源                  |
| -------------------------------- | ------------------------------ | ------------------------- |
| **(A) 小 RL baseline**           | <1M 参数小策略,PPO,PyTorch MPS | 从 reward 学习            |
| **(B) 裸 LLM agent**             | LLM,无记忆                     | 仅当场推理                |
| **(C) LLM + Reflexion + 技能库** | LLM + verbal RL                | 跨 episode 反思与技能复用 |

**为什么必须有 (A):** 小 RL baseline 不是跑龙套,而是**论证的一部分**——"我们买不起 world model / 大 RL,所以用 gridworld + 小策略当 principled surrogate;而 verbal-RL 的 LLM agent 在同样廉价的后端上展现了另一种学习范式。"三方对比让海报从"软件演示"升级为"有对照的实验"。

**指标:** 成功率、救人所耗步数、每 episode 的 regenerate 次数、各条件随 episode 的成功率曲线。

**关键可视化:** 一张 **generate → fail → regenerate → success** 的时间线图(海报杀手锏);一张三条件成功率对比图。

---

## 7. 技术栈

| 层                  | 选型                                                       | 说明                                                                                                      |
| ------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 环境 / mock 后端    | **Gymnasium + MiniGrid**                                   | 标准 RL API;MiniGrid 易改自定义地图;后端可替换                                                            |
| 智能体大脑(主线)    | **Claude API(Haiku 4.5)**                                  | 动作循环用便宜模型;网格世界步数少,成本极低;离线可缓存                                                     |
| 本地模型(可选加分)  | **MLX / mlx-lm(不要用 Ollama)**                            | 苹果统一内存优化;7–8B Q4 在 24G 上舒适运行;原生支持 LoRA。用于 constrained-decoding 的 PL 演示或本地 LoRA |
| 小 RL baseline      | **PyTorch MPS 后端**                                       | M5 上跑 <1M 参数小策略,数小时级                                                                           |
| 结构化/类型安全输出 | **API:tool-use / JSON schema**;本地:type-directed decoding | 见 §4.2                                                                                                   |

**本地模型纠偏:** 团队机器为 M5 24G。**用 MLX 而非 Ollama**。MLX 为统一内存设计,7–8B 4-bit 模型运行舒适、几乎不 swap;12B Q4 可放;14B 会 OOM。Gold 档的 LoRA 微调可在本机用 mlx-lm 完成,无需 Colab。

---

## 8. 范围分层(MVP / Silver / Gold)

| 档                           | 内容                                                                                                                                          | 说明                             |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 🥉 **MVP**(D1–D6 必须完成)    | MiniGrid 救援地图 + LLM agent(**Mode A**:输出结构化动作)+ 一个 forced-dependency 关卡 + regenerate 闭环                                       | **到这一步已足以撑起整场讲解。** |
| 🥈 **Silver**                 | 升级到 **Mode B**:agent 为造物写 DSL 程序(Code as Policy),失败自动定位+重写;**body 作为 typed spec**;编译期/运行期失败分层;Reflexion + 技能库 | generate/regenerate 的完整形态   |
| 🥇 **Gold**(时间富余任选其一) | ① 第二类造物协作;② 更难关卡;③ 本机 MLX 上做一次小 LoRA(呼应"自我改进");④ 本地模型做 type-directed decoding 的 PL 演示                         | 加分项                           |

> **铁律:MVP 在 D6 之前必须跑通;MVP 没跑通,绝不碰 Silver/Gold。**

**Mode A → Mode B 的关系:** Mode A(agent 直接输出结构化动作)是 D3 的"生死门"验证;Mode B(agent 写 DSL 程序)是 Silver 的升级。若 Silver 时间不足,Mode A + regenerate 闭环本身已能完成 presentation。

## 11. 设计约束与待定决策

### 硬技术约束(不可商量)

1. **真·decode-time 类型安全需要 logit 访问** → 只能用本地模型(MLX);API 仅能给 schema 级结构保证。
2. **任意 Python + exec 必崩** → 必须是 DSL(声明式或过程式皆可,但不能是自由 Python)。
3. **纯逐帧 reactive 与"AI 写控制代码"主题冲突** → 若保留代码生成为核心,必须 program-ahead。
4. **LLM API 不可能梯度学习** → 学习只能靠 Reflexion / 技能库等记忆机制。
5. **本地 14B 在 24G 上 OOM** → 本地上限约 12B 且慢;要稳就 MLX + 7–8B Q4。

### 开放决策(留给学术小组)

1. **造物:声明式 vs 构造式** —— 倾向**构造式/有界构造式**(材料散布、`{carry, drop, build}` 拼装),sim-to-real 论证更强;声明式可作降级。
2. **body 是否做成 typed spec** —— 强烈建议是(对称美 + 发挥 PL 背景 + 编译期/运行期分层),但属设计选择。
3. **是否上本地模型做 constrained-decoding 演示** —— 纯 PL 加分项,非必须;API + schema 已够用。
4. **是否加小 RL baseline 做三方对比** —— 强烈建议(科学性 + 贴总纲),但属范围决策。
5. **DSL 具体原语集、地图难度、episode 数量** —— 实现阶段定。

> DSL 原语示例(供讨论,非定稿):
> `move(n); turn(dir); pick(material); craft(tool, [materials]); deploy(body); assemble_creature(name, body); command(name, program); scan(cell); rescue(id)`

---

# PECEI 文献支持地图：站在前人的肩膀上

> 配套 `ROADMAP.md`。按**项目的概念支柱**组织,每篇标注:支撑 PECEI 哪一环、能直接复用的仓库在哪。
> 所有 arXiv 编号 / 会议 / 作者**均已逐篇核实**(2026-08-06),不是凭记忆给的。
>
> 阅读优先级图例:🌟 **必精读**(D1–D2 完成,直接决定方案);★ **强相关**(related work 必引);○ **备查**(具体技术点查到再看)。

---

## 0. 一图看懂:6 根支柱 → 哪篇是那根柱子的"祖师爷"

| PECEI 支柱(ROADMAP 章节)                 | 这个支柱的奠基/代表作                                    | 为什么是它                                                            |
| ---------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------- |
| 心智 = 控制程序(§3)                      | **Code as Policies** / **ProgPrompt**                    | "LLM 生成可执行代码当策略"的范式鼻祖                                  |
| 生成→执行→重写循环(§4)                   | 🌟 **Voyager**                                            | generate→execute→feedback→rewrite 三件套,你们 Mode B 几乎是它的简化版 |
| 无梯度学习 / verbal RL(§5)               | 🌟 **Reflexion** + **ExpeL**                              | "用语言模拟 RL"的两条已验证路径                                       |
| 故障定位再生(§4.3,PL 强项)               | 🌟 **Self-Debugging**                                     | 回溯自修复范式,直接复用团队 AI4SE 积累                                |
| 类型安全 / compile-time 拦截(§4.2)       | **PICARD** / **Grammar-Aligned Decoding**                | 受控解码保证输出合法                                                  |
| **躯体+心智协同设计(§1 thesis 新意)★★★** | **Karl Sims / Evolution Gym / RoboGrammar / DiffuseBot** | "同时生成 body 和 mind"的整条研究脉络                                 |

---

## 1. 心智引擎:LLM 作为策略 / 代码即动作 (支撑 §3 §4.1)

这一组回答 mentor 最可能的第一问:**"为什么 LLM 写的代码能当机器人的大脑?"**

- 🌟 **「Code as Policies: Language Model Programs for Embodied Control」** — Liang et al., *ICRA 2023* · [arXiv:2209.07753](https://arxiv.org/abs/2209.07753) · [项目页](https://code-as-policies.github.io/)
  > **支撑 §3 §4.1**:你们"Mind = 类型化 DSL 程序"的直接理论来源。CaP 证明了 LLM 生成的代码可以作为机器人的反应式策略程序——**Mode A/B 的合法性根基**。把 CaP 的"自由 Python"换成你们的"typed DSL"即可。

- **「ProgPrompt: Generating Situated Robot Task Plans using LLMs」** — Singh et al.(NVIDIA), *AuRo 2023* · [arXiv:2209.11302](https://arxiv.org/abs/2209.11302) · [progprompt.github.io](https://progprompt.github.io/)
  > **支撑 §3**:比 CaP 更强调 **situated awareness**(把环境状态、可用能力写进 prompt 结构)。你们的"结构化 observation + 失败重入"prompt 设计可直接借鉴它的程序化 prompt 骨架。

- ★ **「Do As I Can, Not As I Say: Grounding Language in Robotic Affordances」(SayCan)** — Ahn et al.(Google), *CoRL 2022* · [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) · [say-can.github.io](https://say-can.github.io/)
  > **支撑 §2 §4.4**:LLM 负责"想"(Say),affordance/value 负责判断"物理上能不能做"(Can)。**forced-dependency 关卡**的逻辑先例——本体进不去,等于 affordance 为 0,逼它造物。可用作"为什么智能体必须依赖造物"的文献背书。

- **「ReAct: Synergizing Reasoning and Acting in Language Models」** — Yao et al., *ICLR 2023* · [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) · [code](https://github.com/ysymyth/ReAct)
  > **支撑 §3**:thought→action→observation 交错循环的根范式。你们的"program-ahead + 感知 + 失败重入"是 ReAct 的非逐帧升级版。related work 必引。

- ○ **「Chain-of-Thought Prompting Elicits Reasoning in LLMs」** — Wei et al., *NeurIPS 2022* · [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
  > **备查**:ToT / ReAct / Reflexion 的共同祖先。海报 related work 树状图的根节点,引一篇交代清楚即可。

---

## 2. 生成→执行→重写:Voyager 谱系 (支撑 §4,Mode B 核心)

- 🌟 **「Voyager: An Open-Ended Embodied Agent with LLMs」** — Wang et al., 2023 · [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) · [code](https://github.com/MineDojo/Voyager)
  > **支撑 §4 §5,这是 D1 必精读的第一篇**。三件套——**自动课程 + 可执行代码生成 + 技能库(失败重写)**——几乎就是你们 Silver 档的完整形态。重点看它怎么把失败喂回去让 LLM 重写代码,以及技能库怎么索引复用。

- ★ **「Ghost in the Minecraft (GITM)」** — Zhu et al., 2023 · [arXiv:2305.17144](https://arxiv.org/abs/2305.17144) · [code](https://github.com/OpenGVLab/GITM)
  > **支撑 §4.4**:LLM + 文本知识 + 记忆做开放式环境的**制造/合成(crafting)**。你们"craft 动作 + 材料合成造物"可对照它的 craft 链。

- ★ **「DEPS: Describe, Explain, Plan and Select」** — Wang et al., *NeurIPS 2023* · [arXiv:2302.01560](https://arxiv.org/abs/2302.01560) · [code](https://github.com/CraftJarvis/MC-Planner)
  > **支撑 §4.3**:它的 **Explain** 模块——执行失败时给出解释再重规划——和你们的"故障定位再生"同构。重点抄它失败→解释→重试的闭环结构。

---

## 3. 无梯度学习:反思、记忆、技能库 = Verbal RL (支撑 §5)

回答 mentor 的核心追问:**"用 API 不可能梯度学习,你的 agent 凭什么越练越好?"** 答案 = 这一组。

- 🌟 **「Reflexion: Language Agents with Verbal Reinforcement Learning」** — Shinn et al., *NeurIPS 2023* · [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) · [code](https://github.com/noahshinn/reflexion)
  > **支撑 §5,D1 必精读第二篇**。verbal RL 的命名出处。失败→写反思→存记忆库→下次检索 top-k 塞 prompt。你们 (C) 条件的反射弧就是它。论文里直接有"反思能跨 episode 提升成功率"的对照实验,可直接拿来当 (C) 的预期效果图。

- ★ **「ExpeL: LLM Agents Are Experiential Learners」** — Zhao et al., *AAAI 2024* · [arXiv:2308.10144](https://arxiv.org/abs/2308.10144) · [code](https://github.com/LeapLabTHU/ExpeL) · [项目页](https://andrewzh112.github.io/expel/)
  > **支撑 §5**:Reflexion 的进阶版——自主收集经验→**抽取出可迁移的自然语言 insight**。如果想让 (C) 条件不只是"记反思"还能"提技能",ExpeL 的 insight 抽取流程可直接套。

- ★ **「Generative Agents: Interactive Simulacra of Human Behavior」** — Park et al., *UIST 2023* · [arXiv:2304.03442](https://arxiv.org/abs/2304.03442) · [code](https://github.com/joonspk-research/generative_agents)
  > **支撑 §5 §10(上下文窗口)**:它的 **memory → reflection → retrieval** 三件套就是你们"检索 top-k 反思、不全量拼接"的工程先例。论文里有完整的记忆压缩与检索机制,直接照搬可解"上下文会不会爆"的质疑。

- ★ **「Self-Refine: Iterative Refinement with Self-Feedback」** — Madaan et al., *NeurIPS 2023* · [arXiv:2303.17651](https://arxiv.org/abs/2303.17651) · [code](https://github.com/madaan/self-refine)
  > **支撑 §4.3 §5**:LLM 自己给反馈再改自己输出的范式。比 Reflexion 更轻量(无外部环境),适合做 Mind 程序的**编译期**自修复对照。

- ○ **「Tree of Thoughts (ToT)」** — Yao et al., *NeurIPS 2023* · [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) · [code](https://github.com/princeton-nlp/tree-of-thought-llm)
  > **备查**:把推理当成树搜索 + 回溯。你们 regenerate 的"回溯到出错节点"可类比,但 ToT 是纯推理、你们是程序执行。

- ○ **「LATS: Language Agent Tree Search」** — Zhou et al., *NeurIPS 2023* · [arXiv:2310.04406](https://arxiv.org/abs/2310.04406)
  > **备查**:ToT + ReAct + MCTS 的合体,带环境反馈的回溯。如果想给 Silver/Gold 加"多分支尝试 + 选最优"的搜索,这是天花板方案。

---

## 4. 故障定位再生:自修复 / 回溯 debug (支撑 §4.3)——团队的 PL/AI4SE 主场

这是 PECEI 最该"发挥团队 type-safe decoding + 回溯 debug 积累"的一环。

- 🌟 **「Teaching Large Language Models to Self-Debug」** — Chen et al., *ICLR 2024* · [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)
  > **支撑 §4.3,D1 必精读第三篇**。Self-Debugging 范式:执行生成的代码→观察结果→**基于执行反馈局部修改**。你们的"运行期失败 → 从轨迹尾部定位 → 只重写出错那处"几乎是它的直接复刻。海报杀手锏图(generate→fail→regenerate→success)可对照它的反馈循环画。

- ★ **「CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing」** — Gou et al., *ICLR 2024* · [arXiv:2305.11738](https://arxiv.org/abs/2305.11738)
  > **支撑 §4.3**:强调**靠外部工具/执行信号**纠错,而非纯自省。和你们"运行期失败喂回"的取向一致。

- ⚠️ **「Large Language Models Cannot Self-Correct Reasoning Yet」** — Huang et al., *ICLR 2024* · [arXiv:2310.01798](https://arxiv.org/abs/2310.01798)
  > **支撑 §5 §10(诚实边界)**:**主动引这一篇反而更可信**。它指出无外部信号的自省会**变差**(把对的改成错的)。你们正好可以说:"正因如此,我们的 regenerate 靠**编译期 checker + 运行期执行轨迹**这两类外部信号,而非空想式自省"——把论文的批评变成对你们设计选择的辩护。

---

## 5. 类型安全与受控生成:compile-time 拦截 (支撑 §4.2,本机 MLX 的 PL 演示)

回答:**"怎么保证 LLM 吐出来的 DSL 一定合法?"** API 走 schema,本地走 logit 级受控解码。

- ★ **「PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding」** — Scholak et al., *EMNLP 2021* · [ACL](https://aclanthology.org/2021.emnlp-main.779/) · [code](https://github.com/ServiceNow/picard)
  > **支撑 §4.2 §11(硬约束)**:受控解码的经典——每一步**拒绝不合法 token**。你们本地 MLX 做 type-directed decoding 的理论原型。Gold 档 PL 演示的根文献。

- ★ **「Grammar-Aligned Decoding for LLMs」** — Park et al., *NeurIPS 2024* · [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2bdc2267c3d7d01523e2e17ac0a754f3-Paper-Conference.pdf)
  > **支撑 §4.2**:比 PICARD 更新,保持模型概率分布的前提下做 grammar 约束。配 PICARD 看,就知道受控解码的演进。

- ○ **「Grammar-Constrained Decoding for Structured NLP Tasks」** — Geng et al. · [arXiv:2305.13971](https://arxiv.org/abs/2305.13971)
  > **备查**:把受控解码推广到任意形式文法。DSL 语法约束的通用方法库。

- **工具栈(非论文,但"同伴的脚步"直接踩)**
  > - **Outlines**([outlines-dev/outlines](https://github.com/outlines-dev/outlines))——JSON schema / 正则 / context-free grammar 受控生成,**直接可用**。
  > - **LMQL**([lmql.ai](https://lmql.ai))——声明式约束查询语言。
  > - **TypeChat**(Microsoft)——把自然语言约束进 typed schema 的轻量工程方案,和你们 typed body-spec 思路一致。
  > - **mlx-lm / MLX**([ml-explore/mlx](https://github.com/ml-explore/mlx))——本机受控解码与 LoRA 的载体;MLX 论文 [arXiv:2406.09971](https://arxiv.org/abs/2406.09971)(Hannun et al., 2024)。

---

## 6. ★★★ 躯体与心智协同设计:body + mind co-authoring (支撑 §1 thesis 的新意所在)

> **这是 PECEI 最该被深挖、也是 ROADMAP §13 还没收录的一组。** 你们的命题"Mind and Body, Both Authored"不是凭空原创——它站在一条 30 年的研究脉络上:**同时生成/进化造物的躯体与大脑**。把这一组摆进 related work,thesis 的原创性反而更稳(你是在一个有传承的命题上加上了"LLM 生成 + 失败再生"的新机制)。

- ★ **「Evolving Virtual Creatures」** — Karl Sims, *SIGGRAPH 1994* (pp. 15–22)
  > **支撑 §1,叙事钩子**。30 年前的奠基工作:用遗传算法**同时进化虚拟生物的躯体(形态图)和大脑(神经控制器)**。这是"Mind and Body, Both Authored"的原点。海报开头引这一篇,直接立住"我们做的是一条老命题的新版本"。
  > 资源:[原片/介绍](https://www.karlsims.com/evolved-virtual-creatures.html)、[背景解读](https://pyimagesearch.com/2022/06/13/karl-sims-evolving-virtual-creatures-1994/)

- ★ **「Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots」** — Bhatia et al., *NeurIPS 2022* · [arXiv:2201.09863](https://arxiv.org/abs/2201.09863) · [code](https://github.com/EvolutionGym/evogym)
  > **支撑 §1 §2**:**body(体素布局)+ mind(控制策略)联合优化**的 benchmark。你们的造物"body = 材料规格 + 导出属性"几乎就是它的体素设计的离散化简化版。**最直接的 1:1 对照**:它进化 body+mind,你们用 LLM generate + regenerate body+mind。强推精读。

- ★ **「RoboGrammar: Graph Grammar for Terrain-Optimized Robot Design」** — Zhao et al., *SIGGRAPH Asia / ACM TOG 39(6), 2020* · [ACM DL](https://dl.acm.org/doi/10.1145/3414685.3417831) · [code](https://github.com/allanzhao/RoboGrammar) · [项目页](https://people.csail.mit.edu/jiex/papers/robogrammar/)
  > **支撑 §1 §4.1**:用**图文法(grammar)生成合法的机器人形态**,再学控制。和你们"typed DSL + body schema 约束生成合法造物"完全同构。可借鉴它"grammar 限定设计空间"的思路来设计你们的 body-spec 约束。

- ★ **「DiffuseBot: Breeding Soft Robots with Diffusion Models」** — Zhang, Wang, Zhang et al., *ICML 2024* · [arXiv:2311.01853](https://arxiv.org/abs/2311.01853) · [项目页](https://tsagkas.github.io/diffusebot/)
  > **支撑 §1**:用**生成模型(扩散)造躯体** + 协同进化找高表现设计。是"用 LLM/生成模型而不是进化算法来 generate body"的近期先例——和你们的取向最近。

- ○ **「SoftZoo: A Soft Robot Co-design Benchmark」** — Wang et al., *ICLR 2023* · [arXiv:2303.09555](https://arxiv.org/abs/2303.09555) · [code](https://github.com/zswang666/softzoo)
  > **支撑 §1**:Evolution Gym 的后继,补全了材料/环境多样性,并分析了**co-design landscape**的三种模式——对你们"为什么 body 和 mind 要分开 regenerate"有理论启发。

- ○ **「The Surprising Creativity of Digital Evolution」** — Lehman et al., *Artificial Life 2020* · [arXiv:1803.03453](https://arxiv.org/abs/1803.03453)
  > **支撑 §1,第二个叙事钩子**。一文集锦了"进化/生成系统如何钻空子、打破设计者假设"的真实案例。对你们极有价值:**环境打破造物假设 → 智能体被迫 regenerate** 的整个戏剧张力,这篇给了大量可讲的故事素材。

---

## 7. 造物与工具制造:forced-dependency / tool-making (支撑 §4.4)

- ★ **「Large Language Models as Tool Makers (LATM)」** — Cai et al., *ICLR 2024* · [arXiv:2305.17126](https://arxiv.org/abs/2305.17126) · [code](https://github.com/ctlllll/LLM-ToolMaker)
  > **支撑 §4.4**:LLM **自己造工具**(写函数)再自己用。和你们"智能体为造物写 mind 程序"高度同构——造物即造工具。注意编号是 **2305.17126**(非网上个别错误流传的 2312)。

- ○ **「Toolformer: Language Models Can Teach Themselves to Use Tools」** — Schick et al., *NeurIPS 2023* · [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
  > **备查**:工具使用的基础范式。你们"造物即工具"是它的升级(不只调用工具,还**生成**工具)。

- ○ **「HuggingGPT: Solving AI Tasks with ChatGPT and its Friends」** — Shen et al., *NeurIPS 2023* · [arXiv:2303.17580](https://arxiv.org/abs/2303.17580)
  > **备查**:LLM 当 controller 调度专家模型。多智能体/子智能体协作的范式参考(Silver 第二类造物协作)。

---

## 8. 等效替代辩护:gridworld 作为 principled surrogate (支撑 §2)

> 这是 mentor 第二可能追问的地方:**"网格世界凭什么等同真实机器人实验?"** 用这一组立论。

- 🌟 **「Minigrid & Miniworld: Modular & Customizable RL Environments」** — Chevalier-Boisvert et al., *NeurIPS 2023 D&B* · [arXiv:2306.13831](https://arxiv.org/abs/2306.13831) · [文档](https://minigrid.farama.org/) · [code](https://github.com/Farama-Foundation/Minigrid)
  > **支撑 §2 §7**:你们的后端本体。引这篇确立"MiniGrid 是公认的、为可迁移与泛化而设计的标准 RL 环境"。注意正式编号是 **2306.13831**。

- ★ **「Gymnasium: A Standard Interface for RL Environments」** — Towers et al., 2024 · [OpenReview](https://openreview.net/forum?id=qPMLvJxtPK)
  > **支撑 §2 §7**:标准 API 的出处。"后端可替换"的接口保障。

- ★ **「ALFWorld: Aligning Text and Embodied Environments」** — Shridhar et al., *ICLR 2021* · [arXiv:2010.03768](https://arxiv.org/abs/2010.03768) · [code](https://github.com/alfworld/alfworld)
  > **支撑 §2**:**最贴近你们架构的先例**——文本世界学策略、同构迁移到具身世界。直接支撑你们"抽象层同构、换后端即可迁移"的论点。

- ★ **「SPRING: Studying the Paper and Reasoning to Play Games」** — Wu et al., *NeurIPS 2023* · [arXiv:2305.15486](https://arxiv.org/abs/2305.15486)
  > **支撑 §2**:**游戏/文本世界作为严肃研究替代品**的范例。它读游戏论文→用文本策略玩游戏,正好支撑你们"网格世界是 principled surrogate 而非玩具"的辩护立场。

- ○ **「BabyAI: A Platform to Study Sample Efficiency of Grounded Language Learning」** — Chevalier-Boisvert et al., 2018 · [arXiv:1810.08272](https://arxiv.org/abs/1810.08272) · [code](https://github.com/mila-iqia/babyai)
  > **支撑 §2**:MiniGrid 的前身,gridworld + 指令学习。若做语言条件任务可对照。

- **可选对照平台(支撑 §2 "换后端即迁移"的现实性)**
  > 真实机器人后端的存在本身就证明你们抽象层同构表成立:AI2-THOR([Kolve et al. 2017, arXiv:1712.05481](https://arxiv.org/abs/1712.05481))、Habitat([Savva et al., ICCV 2019, arXiv:1912.05830](https://arxiv.org/abs/1912.05830))。related work 提一句"换这些后端控制结构不变"即可。

---

## 9. 奖励设计与小 RL baseline 的语境 (支撑 §6)

> (A) 小 RL baseline 不是陪跑——它和这一组一起构成"另一种学习范式"的对照。

- ★ **「Eureka: Human-Level Reward Design via Coding LLMs」** — Ma et al.(NVIDIA), *ICLR 2024* · [arXiv:2310.12931](https://arxiv.org/abs/2310.12931) · [code](https://github.com/eureka-research/eureka)
  > **支撑 §6 §10**:LLM **写奖励函数(代码)**再让 RL 学——和你们"LLM 写 mind 程序"是镜像(一个写 reward,一个写 policy)。海报 related work 可用它做"LLM-as-author"谱系的另一支。

- ○ **「Text2Reward: Reward Shaping with Language Models for RL」** — Xie et al., *ICLR 2024* · [arXiv:2309.11489](https://arxiv.org/abs/2309.11489) · [code](https://github.com/xlang-ai/text2reward)
  > **支撑 §6**:Eureka 的同侧工作,LLM 生成可执行 reward 代码 + 反馈迭代。

- ○ **「Language to Rewards for Robotic Skill Synthesis (L2R)」** — Yu et al.(DeepMind), *CoRL 2023* · [arXiv:2306.08647](https://arxiv.org/abs/2306.08647) · [code](https://github.com/google-deepmind/language_to_reward_2023)
  > **支撑 §6**:语言→reward 参数的范式。

- ○ **「Proximal Policy Optimization Algorithms (PPO)」** — Schulman et al.(OpenAI), 2017 · [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
  > **支撑 §6**:你们 (A) 小 RL baseline 的算法出处。引一篇交代清楚。

---

## 10. 前沿/货币性:近期 embodied LLM agent (让海报"很新")

- ○ **「JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal LLMs」** — Wang et al., *NeurIPS 2023* · [arXiv:2311.05997](https://arxiv.org/abs/2311.05997) · [项目页](https://craftjarvis-jarvis1.github.io/)
  > **支撑 §5**:Voyager 之后最全能的 Minecraft agent,**多模态记忆 + 计划 + 技能**。展示 verbal-RL 的当前上限,也说明你们的方法在更大语境里仍成立。

- ○ **「STEVE-1: A Generative Model for Text-to-Behavior in Minecraft」** — Lifshitz et al., *NeurIPS 2023* · [arXiv:2306.00937](https://arxiv.org/abs/2306.00937) · [code](https://github.com/Shalev-Lifshitz/STEVE-1)
  > **支撑 §6**:仅 **$60 算力**训出的指令跟随 agent——支撑你们"廉价后端 + 小投入也能做出像样 agent"的叙事。注意作者名是 **Lifshitz**。

- ○ **「RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation」** — Wang et al., *ICML 2024* · [arXiv:2311.01455](https://arxiv.org/abs/2311.01455) · [code](https://github.com/Genesis-Embodied-AI/RoboGen)
  > **支撑 §1 §4**:用生成式仿真**自动提出任务、生成环境、生成机器人设计**——是"LLM 自动 generate 造物与任务"在机器人侧的近期对照。

---

## 11. 总入口:综述 (一篇文章把 related work 框架搭起来)

- ★ **「The Rise and Potential of Large Language Model Based Agents: A Survey」** — Xi et al., 2023(Science China Info. Sci. 2025 cover paper) · [arXiv:2309.07864](https://arxiv.org/abs/2309.07864) · [配套 paper list](https://github.com/WooooDyy/LLM-Agent-Paper-List)
  > **D1 先读这一篇的目录与图**。它把 agent 拆成 perception / memory / planning / action 四块,正好对你们 §3 架构。related work 的组织框架直接抄它。
- ○ **「A Survey on Large Language Model based Autonomous Agents」** — Wang et al., 2023 · [arXiv:2308.11432](https://arxiv.org/abs/2308.11432)
  > **备查**:姊妹综述,补角度。

---

## 12. 海报 related work 怎么串(一条可直接讲的故事线)

把上面 11 组串成一条叙事,别平铺直叙:

> "Mind and Body, Both Authored" 是一条**有 30 年传承的命题**——从 Karl Sims 同时进化躯体与大脑 [Sims 1994]、到 Evolution Gym / RoboGrammar / DiffuseBot 的 body+mind 联合设计 [Bhatia 2022; Zhao 2020; Zhang 2024]。
>
> 这条命题在 LLM 时代获得新引擎:LLM 生成可执行代码当心智 [Code as Policies; ProgPrompt],并通过 生成→失败→重写 的闭环自我改进 [Voyager; Self-Debugging; Reflexion; ExpeL]——且这种"学习"无需梯度,靠语言反思与技能库实现"verbal RL"。
>
> 我们的工作处在这两条线的交点:**用 LLM 同时 generate 并 regenerate 造物的躯体(类型化材料规格)与心智(类型化 DSL 程序)**,并以编译期/运行期双层失败处理约束这一过程。由于硬件受限,我们用 gridworld 作为可迁移的 principled surrogate [Minigrid; ALFWorld],与小 RL baseline [PPO]、裸 LLM、verbal-RL agent 三方对照。

---

## 13. 复用清单:可直接踩的"同伴的脚步"(代码/环境)

| 你要做的                                 | 直接复用                                                                                                                                                                 |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 自定义救援 gridworld                     | [Farama-Foundation/Minigrid](https://github.com/Farama-Foundation/Minigrid) 的自定义 env 示例                                                                            |
| Voyager 式 generate→execute→rewrite 闭环 | [MineDojo/Voyager](https://github.com/MineDojo/Voyager) 的 action 循环 + 技能库                                                                                          |
| Reflexion 记忆/反思                      | [noahshinn/reflexion](https://github.com/noahshinn/reflexion)                                                                                                            |
| ExpeL insight 抽取                       | [LeapLabTHU/ExpeL](https://github.com/LeapLabTHU/ExpeL)                                                                                                                  |
| Generative Agents 记忆检索               | [joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)                                                                              |
| 本机受控解码 / LoRA                      | [outlines-dev/outlines](https://github.com/outlines-dev/outlines) · [ml-explore/mlx](https://github.com/ml-explore/mlx) · [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| body+mind 联合优化(对照)                 | [EvolutionGym/evogym](https://github.com/EvolutionGym/evogym) · [zswang666/softzoo](https://github.com/zswang666/softzoo)                                                |

---

*核实记录(2026-08-06):全部 arXiv 编号经 web 检索逐篇核对。期间纠正了几处常见误传——LATM = 2305.17126(非 2312.xxxxx)、Text2Reward = 2309.11489(非 2310)、Language-to-Reward = 2306.08647、Minigrid&Miniworld 正式版 = 2306.13831、STEVE-1 作者 Lifshitz、DiffuseBot 无 "Xie" 作者(实为 Zhang et al.)。*