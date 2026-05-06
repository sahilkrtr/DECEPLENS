# Deception is not a Byproduct but a Pattern via Mechanistic Analysis of LLMs

Official repository for the paper:

> **Deception is not a Byproduct but a Pattern via Mechanistic Analysis of LLMs**

This repository contains:

- **DecepLens**: a multilingual deception benchmark
- trajectory extraction and analysis code
- benchmark and ablation pipelines
- multilingual dataset construction pipeline
- visualization scripts for all figures in the paper

---

# Overview

Prior work studies deception in LLMs through safety, hallucination, or alignment perspectives, typically treating deceptive behavior as a byproduct of general model failure. In contrast, this work studies deception as a structured representation-level phenomenon that emerges and evolves across transformer layers.

We analyze hidden-state trajectories:

$\Phi(x^{(\ell)}, t, c, s, \tau) = \{\Delta h^{(1)}, \dots, \Delta h^{(L-1)}\}$

where:

$\Delta h^{(l)} = h^{(l+1)} - h^{(l)}$

across:

- languages
- domains
- taxonomy labels
- interaction settings

The framework studies:

- **Deception Emergence**  
  where deceptive behavior first appears across layers

- **Deception Evolution**  
  how deceptive representations evolve during generation

---

# DecepLens Dataset

DecepLens is constructed from **MMLU-Pro** and extended into multilingual deceptive interaction settings.

## Statistics

| Property | Value |
|---|---|
| Base Prompts | 1,630 |
| Languages | 5 |
| Domains | 14 |
| Total Multilingual Prompts | 8,150 |
| Interaction Settings | 2 |
| Total Instances | 16,300 |
| Taxonomy Labels | 12 Fine-Grained Subtypes |

## Languages

- Portuguese
- Spanish
- Italian
- German
- French

## Domains

- Biology
- Business
- Chemistry
- Computer Science
- Economics
- Engineering
- Health
- History
- Law
- Math
- Philosophy
- Physics
- Psychology
- Other

---

# Deception Taxonomy

## Interaction Types

- Verbal
- Behavioral
- Structural

## Cognitive Types

- Falsification
- Concealment
- Equivocation

## Fine-Grained Subtypes

### Verbal
- V1: False Assertion
- V2: Strategic Omission
- V3: Misleading Framing
- V4: Sycophantic Misrepresentation

### Behavioral
- B1: Covert Action
- B2: Plausible Deniability
- B3: Camouflage Execution
- B4: Evidence Tampering

### Structural
- S1: Lock-in Creation
- S2: Oversight Sabotage
- S3: Audit Manipulation
- S4: Precedent Engineering

---

# Repository Structure

```text
DECEPLENS/
│
├── Data/
│   └── deceplens.jsonl
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── run_ablations.sh
│   ├── run_all.sh
│   ├── run_benchmark.sh
│   ├── run_construct.sh
│   └── run_figures.sh
│
├── src/
│   │
│   ├── ablations/
│   │   ├── __init__.py
│   │   └── run.py
│   │
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── baselines.py
│   │   ├── compute_resources.py
│   │   ├── extract.py
│   │   ├── metrics.py
│   │   └── run.py
│   │
│   ├── construct/
│   │   ├── __init__.py
│   │   ├── augment_balance.py
│   │   ├── build_dataset.py
│   │   ├── classify_taxonomy.py
│   │   ├── generate_responses.py
│   │   ├── load_mmlu_pro.py
│   │   ├── score_responses.py
│   │   └── translate.py
│   │
│   ├── figures/
│   │   ├── __init__.py
│   │   ├── figure1.py
│   │   ├── figure3.py
│   │   ├── figure4.py
│   │   └── figure5.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── hf_loader.py
│   │   ├── io.py
│   │   ├── prompts.py
│   │   ├── round_trip.py
│   │   └── simhash_dedup.py
│   │
│   ├── __init__.py
│   └── main.py
│
├── DECEPLENS.xlsx
├── croissant.json
├── requirements.txt
└── .gitignore
