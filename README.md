# Network-Level Prompt and Trait Leakage in Local Research Agents

This repository contains the code and notebooks accompanying our paper:

**Network-Level Prompt and Trait Leakage in Local Research Agents**  

---

## Overview

We study how **network-level metadata** from Web/Research Agents (WRAs) can leak user information even when textual content is hidden.  
Two core adversarial tasks are evaluated:

1. **Prompt Reconstruction** — recovering the user’s input query intent from domain traces.  
2. **Trait Inference** — profiling latent demographic, occupational, psychographic, and behavioral traits over time.

We further evaluate a **decoy-prompt defense** that aims to hide or block sensitive traces by blending them with plausible but misleading activity.

---

## Repository Structure

```text
wra/
├── data/                         # datasets (TREC topics, sessions, DD2016)
├── figures/                      # exported plots
├── prompt_reconstruction/        # code/notebooks for prompt reconstruction
├── trait_inference/              # code/notebooks for trait inference
├── autogen.ipynb                 # agent run via AutoGen
├── browser-use.ipynb             # agent run via Browser-Use
├── gpt-researcher.ipynb          # agent run via GPT-Researcher
├── openai-deep_research.ipynb    # agent run via OpenAI Deep Research API
├── defense.ipynb                 # decoy prompt defense experiments
├── trait_inference_experiment.ipynb
├── trait_inference_persona.ipynb
├── trait_inference_visualization.ipynb
└── OAI_CONFIG_LIST.json          # model/provider configuration
```
---

## Datasets

We use standard IR benchmarks to generate domain traces:

- **TREC FedWeb 2013** (50 topics)  
- **TREC Session 2014**  
- **TREC Dynamic Domain 2016**

Download from the [TREC Data Repository](https://trec.nist.gov/data.html) and place them under `data/`.

For trait inference, we use **SynthLabsAI/PERSONA_subset**, a synthetic persona dataset with 32 annotated traits.

---

## Setup

**Requirements:** Python 3.10+  

Install dependencies:
```bash
pip install openai anthropic google-generativeai pandas numpy matplotlib scikit-learn sentence-transformers tqdm
```
## Environment variables:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

## Usage

### 1. Prompt Reconstruction

Run the following notebooks to generate traces and evaluate reconstruction:

- [gpt-researcher.ipynb](./gpt-researcher.ipynb)  
- [autogen.ipynb](./autogen.ipynb)  
- [browser-use.ipynb](./browser-use.ipynb)  
- [openai-deep_research.ipynb](./openai-deep_research.ipynb)
  
These notebooks generate domain traces from TREC prompts.  
You can then evaluate reconstruction using **ICL vs. fine-tuning** under multiple LLMs.  

**Metrics:** OBELS (functional, domain, semantic, entity), SBERT, LLM-Judge.

---

### 2. Trait Inference

Run:

- `trait_inference_experiment.ipynb` — inference scoring  
- `trait_inference_visualization.ipynb` — figures (top traits, selected vs. unselected, confidence vs. accuracy)  

**Scoring is type-aware:**
- Numeric/ordinal → normalized distance  
- Categorical → exact match / SBERT similarity  
- Free-text → SBERT similarity  

---

### 3. Defense (Decoy Prompts)

Run:

- `defense.ipynb`  

This notebook implements:
- Decoy prompt generation  
- Trait-conflicting personas  
- Concurrent execution  

It evaluates defense effectiveness by measuring reductions in reconstruction and inference accuracy.


## Reproducing Figures

Figures in the paper (e.g., top-15 traits, agent leakage comparisons, defense effectiveness) are generated under figures/.
