# The Initiative Gap
### A Framework for Autonomous Discovery and Empirical Alignment

> *"Intelligence is not just the ability to answer questions. It is the ability to identify where the answers end and the truth begins."*

---

## What Is This?

Every AI system ever built shares one property: it cannot do anything without being told to.

Give a language model a prompt — it responds. Give it nothing — it does nothing. Between interactions, it is completely dormant. It is not thinking. It is not wondering. It is not curious. It is waiting.

Humans are not like this. We sit quietly and questions arise. We notice gaps in what we know without being asked to notice them. We pursue understanding without being given a task. This capacity — spontaneous inquiry — is what drives scientific discovery. It is what allows us to find questions nobody has thought to ask yet.

No AI system has this. I call the absence of it **the Initiative Gap**.

This repository presents:

1. **A formal definition** of the Initiative Gap and proof that all current architectures exhibit it
2. **The MAIL Framework** (Meta-cognitive Autonomous Inquiry Loop) — a complete architectural specification for a system that closes the gap
3. **The Knowledge Frontier Problem** — what should an AI do when it reaches the edge of human knowledge and receives the answer "I don't know"?
4. **The Competitive Knowledge Arena (CKA)** — a multi-agent extension where AI agents compete under zero-sum epistemic rewards, accelerating discovery through competition
5. **A working prototype** — code implementing the curiosity-driven learning loop on a small model
6. **A proposed experimental programme** for testing these ideas at frontier scale

---

## The Core Ideas

### The Initiative Gap
Current AI systems require external input to produce any output. This is not a minor limitation — it is a structural property of every architecture in deployment. Transformers, diffusion models, reinforcement learning agents — all dormant without a trigger. A system that cannot generate a question without being asked to generate a question cannot discover anything genuinely new.

### The MAIL Framework
MAIL is a five-stage autonomous loop:

```
1. DETECT    →  Find sparse regions in the model's embedding space (knowledge gaps)
2. GENERATE  →  Compute a hypothesis vector from surrounding known vectors
3. QUERY     →  Surface the top-10 most similar and top-10 most opposite concepts to a human
4. VALIDATE  →  If human knows: confirm. If human doesn't know: run physical simulation
5. INTEGRATE →  Write confirmed knowledge into model weights via Elastic Knowledge Consolidation
6. LOOP      →  Return to step 1. The frontier moves outward.
```

### The Knowledge Frontier
A system using MAIL will eventually ask questions no human can answer. When the human oracle says "I don't know," the system transitions to self-directed empirical investigation — building environments, running physics simulations, comparing actual results to predicted hypothesis vectors. It learns from reality rather than from human testimony.

When no simulator calibration can resolve a discrepancy, the system has found a **Discrepancy Frontier** — evidence of an unknown physical force or property. This becomes the highest priority target for new inquiry.

### The Competitive Knowledge Arena
Multiple MAIL agents share a sandbox. Every time one agent confirms a hypothesis, it gains. Every other agent loses by an equal total amount. Zero-sum epistemic competition. The prediction: competition accelerates knowledge acquisition, creates strategic question selection, and eventually produces adversarial gap targeting — agents racing to confirm hypotheses before competitors do. This mirrors the competitive dynamics of human scientific research.

---

## The Mathematics

### Gap Detection
Local density in embedding space:

```
ρ(x) = 1 / d_k(x)
```

Where `d_k(x)` is the average cosine distance to the k nearest neighbours. Low density = gap.

### Hypothesis Generation
Similarity-weighted vector interpolation:

```
wᵢ = cos(vᵢ, G_direction)
wᵢ = wᵢ / Σⱼ |wⱼ|          (normalise)
H  = Σᵢ wᵢ × vᵢ             (hypothesis vector)
```

Vectors similar to the gap pull the hypothesis toward them. Opposite vectors push it away. The 180-degree intuition: if "bad" is at -B, "good" is estimated near +B, modulated by all surrounding vectors.

### Elastic Knowledge Consolidation
Integrating new knowledge without destroying existing knowledge:

```
L_EKC = L_new(θ) + λ × Σᵢ Fᵢ × (θᵢ - θ*ᵢ)²
```

Where `Fᵢ` is the Fisher Information — how important each weight is to existing knowledge. Important weights resist change. Unimportant weights update freely.

### Competitive Reward
```
R_i = +ΔI_i                    (agent that confirmed the hypothesis)
R_j = -ΔI_i / (n-1)           (all other agents)
```

Total reward across all agents sums to zero at every timestep.

---

## Repository Structure

```
initiative-gap/
├── README.md                    ← you are here
├── initiative_gap.docx      ← full research paper (15 sections)
├── curious_ai.py             ← working prototype (Python, runs on Google Colab)
└── LICENCE.txt                  ← CC BY 4.0
```

---

## The Implementation

`curious_ai.py` is an unvalidated implementation of the MAIL loop architecture, designed for testing at scale. It has not yet been run on hardware capable of producing meaningful results — that is precisely what the collaboration request in Section 15 of the paper is seeking.

The code implements:

- A small transformer architecture trained only on English language structure (no world knowledge)
- Phase 1: Reinforcement learning loop where the model is rewarded for generating yes/no questions with no input given
- Phase 2: Interactive binary feedback loop — you answer its questions with yes/no only
- Physics environment via PyBullet for post-frontier empirical testing

**The architecture is designed for, but not yet validated on, frontier-scale models and GPU clusters.** The small model here demonstrates the loop structure only. Meaningful results — gap detection, hypothesis vector confirmation, EKC integration, and the Competitive Knowledge Arena — require the infrastructure described in Section 15 of the paper.

**To run the small-scale implementation on Google Colab:**
```python
!pip install torch pybullet numpy
# Upload curious_ai.py then run:
exec(open('curious_ai.py').read())
```

---

## Collaboration Request

I am 14 (At the time of making this repo at 27/02/2026). I do not have access to a GPU cluster or a frontier model's weight architecture.

The theoretical framework is complete. The mathematics is sound. The experimental designs are specified in Section 15 of the paper. What is missing is compute and access.

I am seeking collaboration with any frontier AI laboratory — **Anthropic, Google DeepMind, OpenAI, Meta AI** — willing to provide:

- Read/write access to a frontier model's embedding layer and attention weight matrices
- Infrastructure for online EKC weight updates between inference calls
- Multi-agent simulation environment for running the Competitive Knowledge Arena at scale
- Mentorship from researchers in continual learning and multi-agent RL

In exchange: full collaboration on experimental design and analysis, and co-authorship on any resulting publications.

If the framework is correct, these experiments will demonstrate for the first time that a frontier AI system can acquire genuinely new knowledge through autonomous inquiry — knowledge not present at training time and not provided by human labels. If it is wrong, the experiments will show exactly where and why, which is equally valuable.

**Contact:** 
Name: Harshal Adari
Email: harshal.work.ai64254@gmail.com
Phone: +447909030782

---

## Citation

If you use these ideas, please cite this repository:

```
@misc{initiative_gap_2026,
  title  = {The Initiative Gap: A Framework for Autonomous Discovery and Empirical Alignment},
  year   = {2026},
  url    = {https://github.com/harshalworkai64254-bit/initiative-gap}
}
```

---

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0).
You are free to share and adapt this work for any purpose, including commercial use, as long as you give appropriate credit.

See `LICENCE.txt` for full terms.
