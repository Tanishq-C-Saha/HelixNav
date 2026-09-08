# HelixNav — Pitch Deck Outline (15 Slides)

---

## Slide 1: Title
**HelixNav**
Hierarchical Reinforcement Learning for Autonomous Quadruped Navigation

- Tanishq, Harsimran Singh Dalal, Ishaan Sharma, Eliza Arora, Samiksha
- BE Fourth Year | CPG 190
- Mentor: Dr. Sachin Kansal
- CSED, Thapar Institute of Engineering & Technology

---

## Slide 2: The Problem
**Classical navigation breaks down when things get messy.**

- Nav2-style stacks (A* + DWA) work great in static, structured spaces: ~100% success
- Success drops as environments get dynamic: 96% (low-density), 88% (high-density)
- Root issue: DWA has no memory, no prediction, no learning — it just reacts
- Real industrial floors have moving people, forklifts, tight corridors — this is where it fails

---

## Slide 3: Why It Fails
**The local planner is the weak link, not the global planner.**

- A* (global planning) is already solved — no need to reinvent it
- The unsolved problem is *local, reactive* navigation in clutter and motion
- Classical local planners can't distinguish situations needing different behavior
- This is exactly where learning-based approaches make sense

---

## Slide 4: Our Idea
**Don't replace what works. Learn only what's broken.**

- Three-layer hierarchy:
  1. **A\*** — global path planning (classical, optimal)
  2. **Learned navigation policy** — local, reactive, trained via RL
  3. **Frozen locomotion policy** — converts commands to leg motion (pre-trained, untouched)
- RL is used only where it clearly beats the classical alternative

---

## Slide 5: System Architecture
*(Insert Figure 2 — Three-Layer Hierarchical Architecture)*

- Top-down: A* → lookahead vectors → navigation policy → velocity commands → locomotion policy → joint targets
- Bottom-up: depth camera + robot state → observations feed the navigation policy
- Clean interface (`PreTrainedPolicyActionCfg`) keeps locomotion layer untouched

---

## Slide 6: A Key Design Choice — Raw Depth, Not Flat Maps
**Why we skip 2D occupancy grids for the local policy.**

- 2D projections throw away height — ramps, cables, low obstacles all look the same
- Raw depth images (96×54, matched to Intel RealSense D435i) preserve full 3D structure
- Sets up future terrain-aware capabilities without redesigning the pipeline

---

## Slide 7: Another Key Choice — Distance-Based Path Encoding
**How the robot "reads" the path ahead.**

- Path given as 8 fixed-distance lookahead vectors (0.5m to 8m ahead)
- Near points = fine obstacle avoidance detail; far points = coarse direction
- Different from attention-based waypoint encoding used in recent ETH RSL work — an open design question we're testing

---

## Slide 8: Where We Sit vs. Prior Work
*(Simple comparison table)*

| Approach | Global Plan | Local Policy | Locomotion |
|---|---|---|---|
| Nav2 (classical) | A* | DWA (rule-based) | Standard controller |
| Haro et al. 2026 (ETH RSL) | Learned | Attention-based, end-to-end | Trained jointly |
| **HelixNav** | **A\*** | **Depth CNN + lookahead, RL** | **Frozen, modular** |

- Closest comparable work: Haro et al. (2026) — we differ in encoding and modularity, not in ambition

---

## Slide 9: What We've Built So Far (CP1–CP6)
**Six checkpoints, all complete.**

- Working Isaac Lab simulation with the Go2 robot
- Occupancy grid + A* global planner, verified
- Depth camera perception pipeline (96×54, D435i-matched)
- Full hierarchical policy composition wired end-to-end
- First trained navigation policy — functional, but with known issues (next slide)

---

## Slide 10: What Went Wrong (and What We Learned)
**CP6 surfaced three honest failure modes.**

- **Circling** near obstacles — too little path context
- **Stalling** near the goal — reward could be gamed without finishing
- **Drift** in open space — reward shaping created shortcuts

*(Insert Figure 7 — CP6 failure trajectories)*

- Each traced to a specific, fixable cause — not a dead end, a diagnosis

---

## Slide 11: The Fix — CP7 Redesign
**Give the policy more context, give the reward less room to cheat.**

- Expanded observations (41 scalar features + depth image + goal flags)
- Reward rebuilt using **Potential-Based Reward Shaping** (Ng et al., 1999) — mathematically can't be gamed
- Added SRU-GRU recurrent memory for tracking motion over time
- Environment rebuilt, debugged, and currently booting for first training run

---

## Slide 12: Engineering Reality Check
**A few honest lessons from working inside Isaac Lab.**

- Sparse framework docs → several bugs only found by reading source code directly
- Examples: silent contact-sensor failure, misinterpreted action dimension, reset race conditions
- Each one would have silently corrupted training if left unnoticed
- This is real infrastructure work, not just running an existing tutorial

---

## Slide 13: Roadmap
**Where this goes next.**

| Checkpoint | Goal |
|---|---|
| CP7 (in progress) | Train with new reward + observations |
| CP8 (planned) | Handle moving obstacles (pedestrian speeds) |
| CP9 (planned) | Deploy on physical Go2 hardware |

- Success target: 80% goal-reach in static clutter, 70%+ with dynamic obstacles

---

## Slide 14: Why It Matters
**Practical value beyond the lab.**

- Enables inspection in hazardous spaces (chemical storage, mining, construction) without risking people
- Legged robots handle uneven terrain without ramps/lifts — lower infrastructure need than wheeled robots
- Modular design (frozen locomotion) means this scales toward multi-robot systems later, without retraining from scratch

---

## Slide 15: Summary
**HelixNav in one line:**
A hierarchical navigation system that uses classical planning where it works and learning only where it's needed — built, debugged, and being trained checkpoint by checkpoint.

- 6 of 9 checkpoints complete
- Clear, diagnosed path to the remaining objectives
- Questions welcome

---

### Notes for building the PPT
- Keep text per slide light — the table/architecture slides (5, 8, 13) are natural spots for a diagram or table instead of paragraphs
- Figures already in your report to reuse: Fig 1 (Go2 in sim), Fig 2 (architecture), Fig 4 (A* path), Fig 5 (depth image), Fig 7 (failure modes), Fig 9 (SRU-GRU backbone)
- Tone throughout: confident but grounded — you're mid-project, not claiming a finished breakthrough