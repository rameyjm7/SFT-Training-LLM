# SFT-Training-LLM — Supervised Fine-Tuning of Qwen2.5-3B-Instruct

**Course:** ECE5554 – Trustworthy Machine Learning | Fall 2025  
**Team Members:** Mack Werner, Gary Ruppert, BJ Janson, Jacob Ramey*  
*Equal Contributors

See our final report here: https://github.com/rameyjm7/SFT-Training-LLM/blob/main/SFT%20Training%20LLM%20Project%20Report.pdf

---

## Overview

This repository contains all code, configuration files, and analysis for **supervised fine-tuning (SFT)** of the **Qwen2.5-3B-Instruct** large language model to enhance reasoning performance.  
The project demonstrates the full workflow from **baseline evaluation** through **fine-tuning**, followed by **quantitative comparison** across multiple reasoning benchmarks.

Our focus was to strengthen *multi-step mathematical and logical reasoning* without degrading the model’s general knowledge or code generation performance.

---

## Objectives

- Evaluate the base **Qwen2.5-3B-Instruct** model on reasoning benchmarks using `lighteval`.
- Fine-tune the model using the **AceReason-1.1-SFT** dataset.
- Analyze improvements across **AIME**, **Math-500**, **GPQA-Diamond**, **CodeGen**, and **MMLU**.
- Experiment with an **advanced data selection strategy (LIMOPro)** for dataset optimization.

---

## Baseline Configuration

The baseline model was evaluated using the following key parameters:

```bash
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
DTYPE="bfloat16"
MAX_MODEL_LENGTH=32768
TEMPERATURE=0.6
TOP_P=0.95
GPU_MEMORY_UTILIZATION=0.95
```

### Evaluated Benchmarks
- **AIME-24 / AIME-25**
- **Math-500**
- **GPQA-Diamond**
- **Codegen (LCB)**
- **MMLU (Redux 2)**

---

## Baseline Results

| Task | Metric | Score ± stderr |
|------|---------|----------------|
| AIME-24 | pass@k_with_k | 0.067 ± 0.046 |
| AIME-25 | pass@k_with_k | 0.033 ± 0.033 |
| Codegen | codegen_pass@1:16 | 0.104 ± 0.019 |
| GPQA-Diamond | gpqa_pass@k_with_k | 0.354 ± 0.034 |
| Math-500 | pass@k_with_k_n | 0.668 ± 0.021 |
| MMLU (Redux 2 Avg) | acc | 0.639 ± 0.046 |

---

## Fine-Tuning Configuration

- **Dataset:** AceReason-1.1-SFT (15,000 samples)
- **Epochs:** 7  
- **Cutoff Length:** 16,384 tokens  
- **Method:** Full SFT fine-tuning  
- **Training Loss:** 0.08976  
- **Training Steps/sec:** 0.276  
- **Samples/sec:** 4.414  

The fine-tuning process was executed via [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) with HuggingFace `transformers` and `trl` integration.

---

## Fine-Tuned Evaluation Results

| Task | Metric | Score ± stderr |
|------|---------|----------------|
| AIME-24 | pass@k_with_k | **0.100 ± 0.056** |
| AIME-25 | pass@k_with_k | **0.120 ± 0.056** |
| Codegen | codegen_pass@1:16 | **0.097 ± 0.018** |
| GPQA-Diamond | gpqa_pass@k_with_k | **0.313 ± 0.033** |
| Math-500 | pass@k_with_k_n | **0.698 ± 0.021** |
| MMLU (Redux 2 Avg) | acc | **0.644 ± 0.048** |

---

## Comparison of Results

| Task | Baseline | Fine-Tuned | Δ (Change) |
|------|-----------|-------------|-------------|
| AIME-24 | 6.7% | **10.0%** | **+3.3 pp** |
| AIME-25 | 3.3% | **12.0%** | **+8.7 pp** |
| Codegen | 10.4% | **9.7%** | −0.7 pp |
| GPQA-Diamond | 35.4% | **31.3%** | −4.1 pp |
| Math-500 | 66.8% | **69.8%** | **+3.0 pp** |
| MMLU (Redux 2 Avg) | 63.9% | **64.4%** | **+0.5 pp** |

**Key Insights:**
- **Mathematical reasoning** improved significantly (+3–9 pp).  
- **General knowledge (MMLU)** improved slightly (+0.5 pp).  
- **GPQA-Diamond** decreased slightly (−4 pp).  
- **Code generation** remained stable.  
- Overall fine-tuning yielded enhanced *reasoning* performance without overfitting or knowledge loss.

---

## Advanced Data Selection (LIMOPro)

We explored the **LIMOPro** method (Large-scale Instruction-following Model based on Prompt-response Optimization) to optimize dataset selection.

- Converted JSON dataset for Causal Information Estimation (CIE).
- Computed perplexity-based impact scores for pruning.
- Planned retention of samples in the **50–90% CIE range** (approx. 1,000–2,000 examples).

Despite partial implementation, resource limitations prevented final integration. Future iterations can leverage LIMOPro to achieve a smaller, more impactful dataset subset.

---

## Conclusion

The supervised fine-tuning of Qwen2.5-3B-Instruct with the AceReason dataset improved reasoning and math performance while maintaining general and coding capabilities.  
Although advanced data selection (LIMOPro) was not fully realized, the results confirm the potential of targeted SFT for improving reasoning-specific performance in mid-sized LLMs.

---

## Links

- **GitHub Repository:** [rameyjm7/SFT-Training-LLM](https://github.com/rameyjm7/SFT-Training-LLM)  
- **HuggingFace Models:** [BJJ5555/ECE6514_models](https://huggingface.co/BJJ5555/ECE6514_models/tree/main)  
- **Baseline Tools:** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)  
- **Reference Implementation:** [reds-lab/Project-Reasoning-SFT-LLM](https://github.com/reds-lab/Project-Reasoning-SFT-LLM)

---

### References

- [GeeksForGeeks: Supervised Fine-Tuning (SFT) for LLMs](https://www.geeksforgeeks.org/artificial-intelligence/supervised-fine-tuning-sft-for-llms/)  
- [HuggingFace TRL: SFT Trainer](https://huggingface.co/docs/trl/main/en/sft_trainer)  
- [DeepWiki: LLaMA-Factory Documentation](https://deepwiki.com/hiyouga/LLaMA-Factory)
