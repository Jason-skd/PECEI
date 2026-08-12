# AIGI — Narrative Logic (English translation)

## Introduction

**Background:** world models, AI twins, embodied intelligence, fine-tuning, LoRA.

Today, humanity is training AIs that each specialize in one domain — one industry, one task at a time.

Our team argues the opposite: **we need only one general-purpose AI.** Like an embryonic stem cell,
it can *differentiate on its own* to produce the specialized intelligence a given task requires.
We call this the **AIGI paradigm — Artificial Intelligence Generated Intelligence.**

Our team focuses on the most special kind of AI — the one that carries physical, bodily meaning:
**embodied intelligence.** We ask a single question:

> *Can a general-purpose AI, through its own generate / regenerate process, produce minds
> specialized for many different scenarios?*

**Relevant academic research (filled in):**

- **Per-domain specialization is the prevailing trend.** World models, digital twins, and
  parameter-efficient fine-tuning (LoRA) all perfect *one* model for *one* domain. AIGI stands in
  deliberate contrast: instead of many specialized models, one general model that specializes itself.
- **Karl Sims, *Evolving Virtual Creatures* (SIGGRAPH 1994)** is the conceptual origin of our
  metaphor — it first showed machines *simultaneously generating a body and a brain*. Our
  "differentiation" framing is the modern, LLM-era heir to that idea.
- **Code as Policies (Liang et al., 2022)** establishes that an LLM can author an *executable
  program* that serves directly as a control policy — the basis for our "mind = generated program."

## Methodology

### A game of **real world**

We deliberately designed a **2D grid world + LLM gradient-free learning** scheme. It answers two
questions.

#### Why can this 2D-grid world substitute for real physical-world research? Because it is real *enough*.
*(Game-construction flowchart to be attached later.)*

1. **The 2D grid is an efficient abstraction of *how intelligence commands hardware*.** The
   primitives of our script — `act()` and `beat()` — are exactly the *intelligence* layer of a real
   embodied agent. A real embodied intelligence does not micro-manage how every joint and motor
   moves; that engineering detail is precisely what our grid world abstracts away.
2. **The environment is realistic enough.** A genuine rescue scenario: in an earthquake, a radiation
   leak, or a wildfire, how does an embodied intelligence complete a rescue mission?

#### How does our AI learn?
*(Prompt-engineering diagram to be attached later.)*

1. **Core architecture: the triple loop** (see ROADMAP).
2. **Buffer-based context compression, complexity-scored experience management — no snowball.**
3. **Long-term memory storage and retrieval as self-evolution — a paradigm not yet implemented.**

## Results

Our prototype demonstrates learning efficacy. A few figures:

**Figure 1 — comparison experiment.** A prototype that has *already learned on other maps* consumes
far fewer **epochs** to explore a complex map than a prototype encountering it fresh; it also spends
fewer **rounds** (it familiarizes itself with the map and rescues the human faster). This shows
genuine learning progress.

**Figure 2 — the final script on the hardest scenario.** We find the LLM fully exploits every keyword
we gave the language and writes complex `if`-`else`, `for`, `while`, and observe→react statements.
The language is **Turing-complete**, which proves our prototype can truly *specialize* under a
specific scenario.

**Figure 3 — supplementary experiment.** The script produced on one map, dropped onto the next map,
actually *cannot run through*. This proves the result is a genuinely *specialized mind* — not just
yet another general mind.

## Conclusion — our learning scheme qualifies as a "prototype"

**The project's real contributions, the real highlights** — because of this paradigm:

1. **Most importantly, it is backend-agnostic.** We use the simplest possible backend — an LLM API.
   But swap in a different intelligence backend (RL, AGI), and this learning paradigm still supports
   an embodied intelligence in targeted specialization.
2. **Our prototype has the ability to self-generate (not yet done) and to regenerate** (seen in
   script correction — corresponding to *regenerating the mind*).
3. **Our prototype truly specializes itself**, yet in specializing it does *not* lose its capacity
   for "re-differentiation." On the contrary — *the process of specialization is itself what drives
   its learning.*

## Discussion

### Several limitations

1. The 2D grid is abstract enough, but it lacks detail, and it cannot prove that this layer of
   abstraction will *definitely* transfer to real embodied intelligence. This, however, is the
   trade-off our team made to grasp the *core* problem within 13 days of summer camp, under
   constrained hardware and compute.
2. We use **epochs and rounds consumed** to show the prototype is genuinely learning; we do not have
   a real benchmark. To our knowledge, the field does not yet have one either — and we welcome the
   professors' guidance.
3. **Our "backend-agnostic" claim is unverified.** We ran only one backend — an LLM API — and never
   swapped in RL (reinforcement learning), supervised learning, or other methods as a controlled
   comparison to show that this learning paradigm is genuinely independent of the backend.
   Constrained by the 13 days of summer camp and limited compute, we could not run such comparative
   experiments.

### The exciting ambitions of this prototype (future outlook)

1. **Digital residents.**
2. **AGEI — Artificial General Embodied Intelligence.**
3. **The ultimate vision: AIGI** — the day AI genuinely creates another AI; the *reproduction* of AI.

**What work, added on top of our prototype, could make these real (filled in):**

- **Outer loop — body regeneration:** when the mind fails repeatedly and the cause is attributed to
  the body's physical limits, generate a *new body spec* (a new morphology) and re-run. This is the
  step from "regenerate the mind" to "regenerate the body," and the seed of *reproduction*.
- **Middle loop — skill atoms + principle distillation + shared memory:** decompose successful
  programs into reusable, composable fragments, and distill cross-task principles offline — giving
  the agent a *growing, transferable* long-term memory.
- **Optional local fine-tuning (LoRA / MLX):** bake the stable, repeatedly-confirmed principles
  back into the base weights — moving from "prompt evolution" toward genuine weight-level learning.

With the inner loop already working, adding the middle loop moves us toward **AGEI**; adding the
outer loop (body regeneration) moves us toward the **AIGI** endgame — AI that authors another AI.

## Acknowledge and reference

...
