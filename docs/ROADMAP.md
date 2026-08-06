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

| 网格世界抽象 | 真实具身对应 | 为何可迁移 |
|---|---|---|
| 离散网格 + 单元占用 | 占用栅格 (occupancy grid) | 本就是真实机器人导航的真实表示 |
| 动作 DSL(`move`/`turn`/`pick`) | 运动原语 (motion primitives) | 真实控制栈即如此分层 |
| 材料 + 属性(防火/防水/浮力) | 物体的 affordance / 材料属性 | 同一套符号推理 |
| 由材料拼装的造物 | 机器人**制造/部署的工具或子智能体** | 真实 tool-use / 可部署子体 |
| 心智 = 控制程序 | 可执行的机器人控制程序 | **1:1**,同一程序换底层 binding 即可驱动机器 |
| generate → fail → regenerate | 造工具+写程序 → 感知失败 → 重造/重写 | 控制结构完全相同 |

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

| artifact | 形态 | 校验时机 | 失效含义 |
|---|---|---|---|
| **Body(躯体)** | 类型化材料规格:`{材料集合, 导出属性(防火/小型/浮力...), 物理约束}` | **编译期**:静态 checker 校验材料兼容性、属性推导、约束满足 | 规格自相矛盾(如"防火但用了易燃材料") |
| **Mind(心智)** | 类型化 DSL 控制程序 | **编译期**:类型/语法/schema 校验;**运行期**:在世界中执行 | 程序逻辑错(撞墙、卡死、超时、造物被烧毁) |

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

| 组成 | 估算 tokens |
|---|---|
| 网格 observation(占用栅格 + 状态) | ~200–300 |
| mind 程序 | ~100–800 |
| 失败轨迹尾部(k≈20 步) | ~1,000 |
| Reflexion 反思 top-5 | ~750 |
| 技能签名 | ~500 |
| **单次决策 prompt 合计** | **~2–4k** |

相对 Haiku 4.5 的 200k 上下文有 50–100 倍余量。**只要用"检索 top-k + 反思摘要"而非全量拼接,上下文就不是技术障碍。**

---

## 6. 实验设计(产出海报证据)

三方对照,在同一后端上比较不同"学习范式":

| 条件 | 智能体 | 学习来源 |
|---|---|---|
| **(A) 小 RL baseline** | <1M 参数小策略,PPO,PyTorch MPS | 从 reward 学习 |
| **(B) 裸 LLM agent** | LLM,无记忆 | 仅当场推理 |
| **(C) LLM + Reflexion + 技能库** | LLM + verbal RL | 跨 episode 反思与技能复用 |

**为什么必须有 (A):** 小 RL baseline 不是跑龙套,而是**论证的一部分**——"我们买不起 world model / 大 RL,所以用 gridworld + 小策略当 principled surrogate;而 verbal-RL 的 LLM agent 在同样廉价的后端上展现了另一种学习范式。"三方对比让海报从"软件演示"升级为"有对照的实验"。

**指标:** 成功率、救人所耗步数、每 episode 的 regenerate 次数、各条件随 episode 的成功率曲线。

**关键可视化:** 一张 **generate → fail → regenerate → success** 的时间线图(海报杀手锏);一张三条件成功率对比图。

---

## 7. 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 环境 / mock 后端 | **Gymnasium + MiniGrid** | 标准 RL API;MiniGrid 易改自定义地图;后端可替换 |
| 智能体大脑(主线) | **Claude API(Haiku 4.5)** | 动作循环用便宜模型;网格世界步数少,成本极低;离线可缓存 |
| 本地模型(可选加分) | **MLX / mlx-lm(不要用 Ollama)** | 苹果统一内存优化;7–8B Q4 在 24G 上舒适运行;原生支持 LoRA。用于 constrained-decoding 的 PL 演示或本地 LoRA |
| 小 RL baseline | **PyTorch MPS 后端** | M5 上跑 <1M 参数小策略,数小时级 |
| 结构化/类型安全输出 | **API:tool-use / JSON schema**;本地:type-directed decoding | 见 §4.2 |

**本地模型纠偏:** 团队机器为 M5 24G。**用 MLX 而非 Ollama**。MLX 为统一内存设计,7–8B 4-bit 模型运行舒适、几乎不 swap;12B Q4 可放;14B 会 OOM。Gold 档的 LoRA 微调可在本机用 mlx-lm 完成,无需 Colab。

---

## 8. 范围分层(MVP / Silver / Gold)

| 档 | 内容 | 说明 |
|---|---|---|
| 🥉 **MVP**(D1–D6 必须完成) | MiniGrid 救援地图 + LLM agent(**Mode A**:输出结构化动作)+ 一个 forced-dependency 关卡 + regenerate 闭环 | **到这一步已足以撑起整场讲解。** |
| 🥈 **Silver** | 升级到 **Mode B**:agent 为造物写 DSL 程序(Code as Policy),失败自动定位+重写;**body 作为 typed spec**;编译期/运行期失败分层;Reflexion + 技能库 | generate/regenerate 的完整形态 |
| 🥇 **Gold**(时间富余任选其一) | ① 第二类造物协作;② 更难关卡;③ 本机 MLX 上做一次小 LoRA(呼应"自我改进");④ 本地模型做 type-directed decoding 的 PL 演示 | 加分项 |

> **铁律:MVP 在 D6 之前必须跑通;MVP 没跑通,绝不碰 Silver/Gold。**

**Mode A → Mode B 的关系:** Mode A(agent 直接输出结构化动作)是 D3 的"生死门"验证;Mode B(agent 写 DSL 程序)是 Silver 的升级。若 Silver 时间不足,Mode A + regenerate 闭环本身已能完成 presentation。

---

## 9. 十天路线图

| 日 | 目标 | 产出 | 门控 |
|---|---|---|---|
| **D1** | 立题 + 文献:钉死 thesis;精读 VOYAGER(generate→execute→feedback→rewrite 循环);跑通 MiniGrid 现成 env 并 render | thesis 一页 + 架构图 | — |
| **D2** | 环境地基:基于 MiniGrid 写最小救援地图(agent + 幸存者 + 一个障碍),先不碰材料,让 agent 能走到幸存者 | 可 render 的自定义 env | — |
| **D3** | LLM agent 跑通(**Mode A**):收文字 observation → 输出结构化 action → env.step,解开一个简单关 | 第一段轨迹视频 | **★ 项目生死门:能跑通则后续全是增量** |
| **D4** | 材料 + 造物(body):加材料系统 + craft 动作;设计第一个 forced-dependency 关卡(本体太大,须造小型防火 scout) | agent 会造物 | — |
| **D5** | Regenerate 闭环(mind):agent 为造物写 DSL 程序(Mode B),执行失败即喂回重写 | generate→fail→regenerate→success 轨迹(海报杀手锏) | — |
| **D6** | 实验 + 对照:跑 (A)/(B)/(C) 三条件,记录成功率/步数/regenerate 次数 | 对比图表 | **★ MVP 锁定** |
| **D7** | Stretch:挑一个 Gold 项 | — | — |
| **D8** | 打磨可视化:generate/regenerate 时间线图;挑戏剧性轨迹视频;写海报正文 | 海报素材齐 | — |
| **D9** | 海报定稿 + 讲稿 + 演练 | 终稿 | — |
| **D10** | 缓冲 + 最终演练 | — | — |

---

## 10. 风险与对冲

| 风险 | 对冲 |
|---|---|
| RL 调参地狱 | **主线根本不训**;LLM 当策略。小 RL 仅作 baseline,且 <1M 参数、网格世界,数小时级 |
| 代码生成不可靠 | **typed DSL + 沙盒执行**;编译期 schema 约束;Mode A 兜底;禁止任意 Python + exec |
| API 成本/延迟 | Haiku 4.5 跑动作循环;网格世界步数少;离线可缓存;无 live demo 门槛 |
| 范围膨胀 | D6 锁 MVP 的铁律;Silver/Gold 仅在 MVP 跑通后触碰 |
| MiniGrid 学习曲线 | 为易用设计,团队 Python 水平足够 |
| 上下文窗口 | 检索 top-k + 反思摘要,不全量拼接(见 §5) |

---

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

## 12. 先读资料(D1–D2,别贪多)

- **VOYAGER**(arXiv [2305.16291](https://arxiv.org/abs/2305.16291))——重点看"生成可执行代码 + 技能库 + 失败重写"三件套。本项目的 Mode B 几乎是它的简化版。
- **Reflexion**(arXiv [2303.11366](https://arxiv.org/abs/2303.11366))——verbal reinforcement learning,直接对应"无梯度学习"。
- **MiniGrid 文档** + 一个 example env(<https://minigrid.farama.org/>)——照着抄一个自定义 env。
- (可选,海报 related work)**Code as Policies**、**Eureka**、**Self-Debugging**(自修复/回溯 debug)。

---

## 13. 关键文献速查

- Voyager — Wang et al., 2023, [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- Reflexion — Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Minigrid & Miniworld — Chevalier-Boisvert et al., NeurIPS 2023
- Gymnasium: A Standard Interface for RL Environments — Towers et al., 2024([OpenReview](https://openreview.net/forum?id=qPMLvJxtPK))
