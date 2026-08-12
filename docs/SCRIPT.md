# AIGI — 8-Minute Talk Script

> For a non-native English speaker. Target **~1,000 spoken words → ~8 min at a 125 wpm pace**.
> Cues in brackets are *not* spoken: `[Slide N]` = advance slide; `[≈M:SS]` = target clock time;
> **bold** = stress this word; *(pron: ...)* = pronunciation hint.

### Pronunciation key (say these aloud once first)
- **AIGI** → say the letters "A-I-G-I", then the full name once: *Artificial Intelligence Generated Intelligence*.
- **embodied** *(pron: em-BOD-eed)* · **paradigm** *(PAIR-uh-dyme)* · **differentiate** *(dif-fren-shee-ate)*
- **Turing-complete** *(TOOR-ing com-PLEET)* · **epochs** *(EP-ux)* · **LoRA** *(LO-ruh)* · **regenerate** *(ree-JEN-er-ate)*

---

## [Slide 1 — Title: AIGI] — `[≈0:00]`

Good afternoon, everyone.

Right now, this is how we build AI. For every job — for every industry — we train a **different**, specialized model. We fine-tune. We use techniques like **LoRA**. One model, one task.

Our team believes the opposite. We believe we need only **one** general AI — and that one AI can **specialize itself**.

Think of an embryonic **stem cell**. It is general. It can become anything. But on its own, it *differentiates* — into a nerve cell, a muscle cell, a blood cell.

We want an AI that does the same. It starts general, and it **generates** the specialized intelligence each task needs. We call this **AIGI — Artificial Intelligence Generated Intelligence**.

And we picked the hardest test for it: **embodied intelligence** — AI that has a physical body. Can one general AI, through its own generate-and-regenerate process, produce minds specialized for many different scenarios? **That is our question.**

`[≈1:25]`

## [Slide 2 — Methodology: a game of real world] — `[≈1:25]`

To answer it, we built a **2D grid world** — a small game — plus an LLM that learns **without any gradients**.

You may ask: *a game? Is that real research?* Yes — because it is real **enough**. Let me explain.

In our game, the agent writes a small program with only two verbs: **act** and **beat**. `act` moves the body. `beat` senses the world. These two verbs are exactly the **intelligence layer** of a real robot. A real robot does not decide how every motor spins — that engineering is what our grid **abstracts away**. As the MiniGrid community has shown, grid worlds are a legitimate testbed for embodied AI.

And the world is a real **rescue scenario** — an earthquake, a radiation leak, a wildfire. The agent must reach a human and save them.

`[≈2:25]`

## [Slide 3 — How the AI learns: the blind-author triple loop] — `[≈2:25]`

So how does it learn? In **three loops**.

**Inner loop:** the LLM writes one complete program — a full "mind" — and we run it **blind**. No feedback while it runs. When it stops, we tell it *what happened*: did it reach the goal, hit a wall, or run out of time? Then it **regenerates** a better program. *Generate, run, feedback, regenerate.*

This has strong roots. **Code as Policies** showed an LLM can write a program that *is* the policy. **Voyager** showed the generate-execute-rewrite loop with a growing skill library. **Reflexion** called learning-from-failure "verbal reinforcement" — learning in **words**, not in weights.

**Middle loop:** across many attempts, we compress each failure into a strict **IF-THEN rule**, kept in a **complexity-scored memory buffer**. We do **not** just pile up history. Hard lessons stay; old ones decay. This keeps the memory sharp.

**Outer loop:** when the mind keeps failing because the **body** is the limit, the agent designs a **new body**. That part is our future work.

`[≈3:55]`

## [Slide 4 — Results, Figure 1: the learning gain] — `[≈3:55]`

Now, the results. Does it actually learn?

**Figure one.** We compare two agents on a complex map. The agent that has **already trained** on simpler maps uses **far fewer epochs**, and many fewer **rounds**, to reach the goal than one that starts fresh. It learned. It transferred experience. **This is real learning progress.**

`[≈4:35]`

## [Slide 5 — Results, Figure 2: the specialized mind] — `[≈4:35]`

**Figure two.** This is the final program our agent wrote for the **hardest** scenario. Look at it. It uses everything we gave the language — `if`-`else`, `for`, `while`, and observe-then-react blocks. The language is **Turing-complete**. This is not a canned script. The agent **composed** a complex, specialized mind, on its own, for **this** exact danger.

`[≈5:15]`

## [Slide 6 — Results, Figure 3: proof of specialization] — `[≈5:15]`

**Figure three** is our favorite. We took that successful program — and we dropped it onto the **next** map. It **failed**. It could not run through.

That failure is the proof. If our agent were just *another* general mind, the program would still work. It doesn't. So the mind it wrote is **genuinely specialized** — shaped to one scenario, not a generic copy.

`[≈5:55]`

## [Slide 7 — Conclusion: a true prototype] — `[≈5:55]`

So what have we built? We call it a **prototype**, and we mean it. Three things make it real.

**One — it is backend-agnostic.** We used a simple LLM API. But swap in reinforcement learning, or a real robot — the learning paradigm **still holds**.

**Two — it can regenerate.** When a program fails, it fixes it. That is *regenerating the mind*. Generating a brand-new mind from scratch is our next step.

**Three — and most beautiful.** It specializes **without losing** its power to specialize again. Specialization is not a dead end. The process of specializing **is** how it learns.

`[≈7:00]`

## [Slide 8 — Discussion: limitations and the future] — `[≈7:00]`

We are honest about limits. Our grid world is abstract — it lacks detail, and we cannot promise every abstraction transfers to real robots. But in **thirteen days**, on limited hardware, this is the trade-off we chose to hit the **core** problem.

We measure learning in **epochs** and **rounds**, because — to our knowledge — there is no real benchmark yet. We welcome the professors' guidance here.

And one honest gap: we call this paradigm **backend-agnostic** — but we only tested **one** backend, the LLM API. We never swapped in reinforcement learning, or supervised learning, to prove the paradigm truly holds across backends. In thirteen days, on limited compute, we could **not** run those controlled experiments.

Looking forward. Our inner loop works. Add the middle loop — shared memory, reusable skills — and we move toward **AGEI**, a general embodied intelligence. Add the outer loop — the agent **regenerating its own body** — and we reach the ultimate vision: **AIGI**. The day AI creates another AI — an idea Karl Sims first imagined thirty years ago, with creatures that grew their own bodies and brains.

**Thank you.**

`[≈8:00]`

---

### Delivery notes
- **Pace:** if you tend to speak fast when nervous, the `[≈M:SS]` markers give ~10 s of slack per slide — breathe there.
- **Hardest line to say:** "it specializes **without losing** its power to specialize again" — practice it three times slowly; it is the talk's punchline.
- **Slide 6 ("it failed")** — say it with a small smile. The failure *is* the evidence; let the audience feel the twist.
- If running long, the safest cut is one sentence in the Discussion's "honest gap" paragraph (the backend-agnostic limitation).
