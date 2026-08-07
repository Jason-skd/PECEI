# Paper Extracts — 索引

> 10 篇核心论文,由 marker 从 `../` 下 PDF 提取为 Markdown。
> 每篇:`<论文名>/<论文名>.md`(正文,含表格/公式)+ 同目录 `_page_*_Figure_*.jpeg`(抽取的图)+ `_meta.json`。
> 图片为相对引用,在本目录内打开 .md 即可正常显示。

## 内循环:Generate / Regenerate + 自我诊断(ROADMAP §4)

| 论文 | md 路径 | arXiv/DOI | 支撑 |
|---|---|---|---|
| **Voyager: An Open-Ended Embodied Agent with LLMs** — Wang et al., NVIDIA/Caltech, 2023 | `Voyager_2023/Voyager_2023.md` | [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) | generate→execute→**失败重写**+ 技能库;Mode B 模板 |
| **Code as Policies** — Liang et al., Google, ICRA 2023 | `CodeAsPolicies_2022/CodeAsPolicies_2022.md` | [arXiv:2209.07753](https://arxiv.org/abs/2209.07753) | Mind = 可执行代码当策略 |
| **Teaching LLMs to Self-Debug** — Chen et al., Google DeepMind, ICLR 2024 | `SelfDebugging_2023/SelfDebugging_2023.md` | [arXiv:2304.05128](https://arxiv.org/abs/2304.05128) | §4.3 故障定位再生,海报杀手锏图原型 |

## 中循环:Verbal RL / 跨 episode "进化"(§5)

| 论文 | md 路径 | arXiv/DOI | 支撑 |
|---|---|---|---|
| **Reflexion: Verbal Reinforcement Learning** — Shinn et al., NeurIPS 2023 | `Reflexion_2023/Reflexion_2023.md` | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) | §5.2 反思,verbal RL 命名出处 |
| **ExpeL: LLM Agents Are Experiential Learners** — Zhao et al., AAAI 2024 | `ExpeL_2023/ExpeL_2023.md` | [arXiv:2308.10144](https://arxiv.org/abs/2308.10144) | §5.2 技能原子 + §5.4 经验蒸馏成可迁移原则 |
| **Generative Agents** — Park et al., Stanford, UIST 2023 | `GenerativeAgents_2023/GenerativeAgents_2023.md` | [arXiv:2304.03442](https://arxiv.org/abs/2304.03442) | memory→reflection→retrieval,§5.3/§5.5 记忆工程 |

## 外循环 + thesis 新意:形态再生 = "繁殖"surrogate(§6,§1)

| 论文 | md 路径 | arXiv/DOI | 支撑 |
|---|---|---|---|
| **Evolution Gym** — Bhatia et al., MIT, NeurIPS 2022 | `EvolutionGym_2022/EvolutionGym_2022.md` | [arXiv:2201.09863](https://arxiv.org/abs/2201.09863) | body+mind 联合优化,外循环 1:1 锚点 |
| **Evolving Virtual Creatures** — Karl Sims, SIGGRAPH 1994 | `Sims_1994_EvolvingCreatures/Sims_1994_EvolvingCreatures.md` | [DOI:10.1145/192161.192167](https://doi.org/10.1145/192161.192167) | "同时生成躯体与大脑"原点 + 生物学隐喻叙事钩子 |

## forced-dependency + 后端辩护(§4.5,§2)

| 论文 | md 路径 | arXiv/DOI | 支撑 |
|---|---|---|---|
| **SayCan** — Ahn et al., Google, CoRL 2022 | `SayCan_2022/SayCan_2022.md` | [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) | affordance="Can",本体进不去=affordance 0 |
| **Minigrid & Miniworld** — Chevalier-Boisvert et al., NeurIPS 2023 | `Minigrid_2023/Minigrid_2023.md` | [arXiv:2306.13831](https://arxiv.org/abs/2306.13831) | 后端本体 + §2 等效替代辩护 |

---

**提取元数据:** marker,surya layout/table 模型,原生数字 PDF 关闭 OCR(Reflexion/SelfDebugging 首跑为全量 OCR;批量复跑统一为文本层抽取)。212 页 / 1076 秒。140 张图全部保留。
