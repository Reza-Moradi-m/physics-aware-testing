# RobustAI: Physics-Aware Reliability Testing for Machine Learning

**Independent research and engineering project | Reza Moradi | 2026**  
**External research advisor:** Leo Lipkin (industry mentor; mentorship was independent of his employer)

> **Project status:** Experimental research prototype, not a production-certified safety system or peer-reviewed publication. This repository contains an early public snapshot; some broader experiments and reporting components described below were developed outside that snapshot and still need to be uploaded and made reproducible.

## Research question

A classifier may perform well on clean evaluation data yet fail to deliver usable decisions when sensor readings are corrupted or results arrive too late. **How should ML systems be evaluated when the conditions of deployment differ from ideal benchmark assumptions?**

RobustAI explores this question through controlled perturbations and deadline-aware evaluation. Rather than treating predictive accuracy as the only measure of success, the project considers **data quality, timing, and operational reliability**.

## What is documented in the public repository

The existing project notes describe:

- A reproducible synthetic binary-classification baseline using **logistic regression**, with a reported clean accuracy of approximately **0.97** in one configuration.
- **Post-training Gaussian noise injection** at evaluation time, with severity controlled by noise standard deviation.
- Accuracy comparisons across different noise levels, automated `pytest` checks, and CSV/plot reporting.
- A **deadline-constrained latency experiment** with a reported failure boundary near a configured **120 ms deadline**. The key distinction is between predictive accuracy and *effective accuracy* (predictions received before the deadline and correct).

These are **results reported in the project documentation**, not independently reproduced as part of this README revision. See [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) for an explicit evidence/limitations discussion.

## Extended prototype scope (author-reported; public reproducibility pending)

In local development, I also worked on comparisons involving logistic regression and neural-network models, additional failure profiles (including delay, dropout, staleness, drift, and quantization), result dashboards, and generated audit summaries. **Do not interpret this list as a claim that every feature is currently present, runnable, or validated in this public GitHub snapshot.** Representative scripts, configurations, generated artifacts, and test outputs should be uploaded to substantiate these claims.

## Why timing matters

For systems with deadlines, a prediction can be *correct but unusable*. One useful definition is:

```
effective_accuracy = count(correct AND on-time predictions) / count(all evaluated predictions)
```

This metric should be reported **alongside** predictive accuracy and deadline-miss rate, because otherwise an apparent performance collapse can be confused with degradation of the underlying model's predictions.

## Reproducing the public baseline

The tracked `requirements.txt` currently lists `numpy`, `scikit-learn`, `pytest`, and `matplotlib`. For a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest
```

> The exact command for generating baseline CSV files and plots should be checked against the current `src/` and `tests/` modules before publication. **This documentation does not claim that the commands above reproduce every reported figure.** If the full local RobustAI prototype needs additional packages (for example PyTorch or an LLM backend), those dependencies should be placed in a separate, tested requirements file.

## Repository roadmap for reproducible research

1. Publish the current code snapshot for each reported experiment and identify the entry point.
2. Save experiment configurations, random seeds, data provenance, and complete environment requirements.
3. Add a baseline-results table and plots with axes, units, sample sizes, and repeated-run variation.
4. Document metric definitions: predictive accuracy, effective accuracy, deadline-miss rate, and robustness thresholds.
5. Upload a sanitized sample dashboard and representative generated audit report.
6. Identify which components were used in synthetic demonstrations versus the NASA C-MAPSS extension.

## Artifacts and background

- [Technical research report and interpretation](RESEARCH_REPORT.md)
- Existing development log in Git history (early repository notes)
- [GitHub profile](https://github.com/Reza-Moradi-m)

## Research integrity statement

This is **independent, non-peer-reviewed student research**, developed by Reza Moradi with external research mentorship from Leo Lipkin. It is not presented as research conducted by Broadcom or as a university lab project. The original public code and results should be used to verify any specific numerical claims before they appear in a publication or formal research statement.
