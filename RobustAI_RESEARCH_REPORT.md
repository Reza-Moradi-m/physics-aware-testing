# RobustAI: Evaluating Machine-Learning Reliability Under Sensor Noise and Timing Constraints

**Technical research report — draft, non-peer-reviewed (2026)**  
**Author and primary developer:** Reza Moradi  
**External research advisor:** Leo Lipkin (industry mentor; independent of his employment)  
**Project repository:** https://github.com/Reza-Moradi-m/physics-aware-testing  
**Research status:** Exploratory prototype; further reproducibility and validation work in progress

## Abstract

Conventional machine-learning evaluation commonly emphasizes predictive accuracy under a fixed, clean test distribution. Systems that interact with sensors, networks, or time-sensitive applications may fail for different reasons: their inputs can be corrupted or delayed, and even correct predictions may arrive too late to be useful. RobustAI is an independent experimental framework intended to test these deployment-sensitive failure modes systematically. The public project documentation reports a synthetic logistic-regression classifier reaching approximately 0.97 baseline accuracy, evaluation-time injection of Gaussian sensor noise, automated testing and visual reporting, and a latency experiment showing a sharp loss of deadline-constrained effectiveness near a configured 120 ms threshold. The broader locally developed prototype has also explored additional perturbations, model comparisons, and audit-style reporting, but the public repository has not yet been reconciled with all of these components. This report describes the motivating problem, documented methodology, preliminary interpretations, major limitations, and a concrete plan for making the research reproducible. It does not claim publication, field validation, or statistically established generality.

## 1. Problem and motivation

A model that classifies 97% of clean observations correctly may not be reliable when deployed with an imperfect sensor or a strict operational deadline. The intended output of a deployed system is not merely a label or a probability; it is a *usable decision* delivered using information that remains sufficiently valid for the task. RobustAI investigates the distinction between predictive quality under ideal assumptions and effective behavior under degraded data and timing conditions.

Three research questions guide the current effort:

1. **Noise sensitivity:** How does classification performance change when evaluation inputs are contaminated by increasingly severe sensor noise?
2. **Timing constraints:** How often is a correct prediction rendered unusable because it is delivered after a specified deadline?
3. **Robustness assessment:** Can consistent failure injection, metrics, automated checks, and reports make performance limitations easier to see before deployment?

The project is best understood as experimental ML reliability testing. The term *physics-aware* refers to approximating constraints such as sensor noise and timing in test conditions; the present synthetic demonstrations do not constitute a physical validation study.

## 2. Project organization and research contributions

RobustAI was independently designed and implemented by Reza Moradi with guidance from external research advisor Leo Lipkin. The mentor is an industry professional, but the project is not claimed as work commissioned, supervised, or sponsored by his employer. No peer-reviewed publication is claimed.

The contributions intended by the prototype are: a clean evaluation baseline; controlled degradation experiments; deadline-aware metrics; automated checks; and reports explaining where predictive or operational behavior becomes unreliable. Additional model adapters, failure profiles, dashboards, and LLM-assisted summaries have been described by the author, but their public reproducibility remains to be demonstrated in the versioned repository.

## 3. Methodology documented in the public repository

### 3.1 Baseline classification

The public development log describes generation of a **synthetic binary classification dataset** and training a **logistic-regression** baseline. The reported accuracy under noise-free conditions is approximately **0.97** for one configuration. This number is a documented example result, not a cross-dataset benchmark, confidence interval, or claimed state-of-the-art result.

A valid reproduction should specify the data-generation procedure, train/test split, sample sizes, random seed, preprocessing, regularization settings, and evaluation code. These details must be verified in the repository before a final paper provides a numerical table.

### 3.2 Noise injection and robustness measurement

In the documented experiment, Gaussian noise is introduced **after training, at evaluation time**, with a tunable standard deviation. Accuracy is then measured at multiple noise severities and logged as tabular outputs and plots. Automated `pytest` checks verify the baseline and selected expected behaviors under mild and strong perturbations.

An important scientific caution: accuracy need not decline *monotonically at every noise step in every finite sample*, because random perturbations and measurement variance can produce reversals. A stronger future protocol should use several random seeds, uncertainty intervals, and trend summaries rather than only deterministic threshold assertions.

### 3.3 Deadline-aware evaluation

The existing README describes a latency-boundary experiment with a deadline near **120 ms**. The document reports that effective accuracy becomes very low as delayed predictions miss this constraint, even while the predictions themselves may remain mathematically correct.

A suitable metric is:

$$
A_{effective} = \frac{\#\{i : \hat y_i = y_i\ \land\ t_i \le D\}}{N}
$$

where `D` is the configured deadline, `t_i` is end-to-end decision latency, and `N` is the total number of evaluated examples. It should be accompanied by standard predictive accuracy and a deadline-miss rate:

$$
M_D = \frac{\#\{i : t_i > D\}}{N}.
$$

This distinguishes an input/model failure from a *system-timing* failure. In particular, a sudden effectiveness drop at the deadline is partly an expected consequence of the chosen operational metric; it is **not, by itself, evidence that the classifier's raw predictive accuracy deteriorated**.

### 3.4 Broader prototype and experimental coverage

Project notes and the author's descriptions include logistic-regression and neural-network comparisons, other simulated conditions (staleness, dropout, bias drift, saturation, quantization, timing variability), and audit-style artifacts including dashboards and summaries. These activities may be meaningful additions, but this report deliberately does **not** present them as publicly reproduced results. The code, experiment configurations, and generated artifacts must be uploaded or linked before exact quantitative claims can be made.

## 4. Preliminary results and evidence ledger

| Observation | Status | Interpretation and caveat |
| --- | --- | --- |
| Approx. 0.97 clean accuracy for a synthetic logistic-regression example | Reported in public README | Illustrative baseline; unknown variance and full configuration without code-level review |
| Accuracy declines under stronger Gaussian noise in described tests | Reported in public README | Evidence of sensitivity in the example setting; not a universal monotonicity law |
| Accuracy-versus-noise plot and CSV/report outputs | Described in public README | Confirm the precise tracked artifact paths and recompute before publication |
| Reported threshold behavior near a configured 120 ms deadline | Reported in public README | Relevant to *deadline-constrained effective accuracy*, not necessarily raw accuracy |
| MLP comparison, extended stress profiles, dashboard, LLM summaries | Author-reported prototype development | Requires publication of updated code/results or explicit labeling as not yet reproducible |
| Generalization to real-world deployments and safety-critical operation | **Not established** | Requires real datasets, workloads, repeated trials, and application-defined success criteria |

## 5. Interpretation and significance

The central lesson is methodological: **clean predictive performance does not fully determine a system's operational usefulness**. Two failure mechanisms should be kept separate. Input corruption can change the model's actual predictions. Timing constraints can make otherwise-correct predictions unusable. A disciplined reliability evaluation reports these mechanisms independently before combining them into a deployment-oriented score or threshold.

This distinction matters in sensor-based monitoring, robotics, industrial telemetry, and other systems with limited tolerance for stale or missing information. Nonetheless, the current experiment is an illustrative prototype and cannot establish safety, resilience, or reliability in any particular production application.

## 6. Limitations and threats to validity

- **Synthetic-to-real gap:** Gaussian noise and simulated delays may poorly approximate correlated failures, distribution shift, sensor calibration errors, real network conditions, or adversarial conditions.
- **Metric sensitivity:** Effective accuracy depends on the chosen deadline. The selected 120 ms threshold is an experimental setting, not a universally justified application requirement.
- **Finite-sample variation:** One accuracy value, one random seed, or one curve cannot support broad statistical conclusions.
- **Causal attribution:** The cause of degradation must be isolated: input corruption, inference delay, network delay, timeout policy, and preprocessing can behave differently.
- **Model coverage:** Demonstrations with logistic regression and neural-network baselines do not generalize automatically to foundation models or embodied agents.
- **Artifact mismatch:** The public repository currently documents an early code snapshot; additional local prototype claims need versioned, runnable supporting evidence.
- **No peer review or external deployment:** Results should be presented as exploratory and preliminary.

## 7. Reproducibility and next experiments

Before promoting this draft to a formal technical paper, the following would substantially strengthen it:

1. **Versioned implementation:** Publish the exact scripts, datasets or generation scripts, dependency versions, and seeds for each table or figure.
2. **Statistical replication:** Repeat perturbation sweeps across several independent seeds, reporting means, dispersion, and confidence intervals.
3. **Model comparisons:** Under a matched data split and perturbation protocol, compare logistic regression and the PyTorch MLP; report accuracy, calibration if relevant, and run-time measurements.
4. **Separate metrics:** Report clean accuracy, stressed predictive accuracy, effective accuracy, deadline-miss rate, and end-to-end latency distribution.
5. **Realistic workloads:** Add a documented NASA C-MAPSS experiment only after data preprocessing, prediction target, evaluation split, and outcome metrics are made public.
6. **Artifacts:** Link a representative dashboard, a generated report, a plot with axis labels and confidence bands, and a short table of reproducible numbers.
7. **Definition of failure:** Set thresholds based on a specific deployment scenario rather than claiming a universal safe/unsafe boundary.
8. **Advisory review:** Ask the external research advisor to review the methodology and the attribution of his mentorship before publishing his name in the repository.

## 8. Conclusion

RobustAI investigates how apparently strong machine-learning performance can become fragile under nonideal input and timing conditions. The documented experiments motivate a clear distinction between predictive accuracy and deadline-constrained effectiveness. The next research milestone is not adding grander claims; it is connecting the full local prototype to a transparent public repository, reproducing numerical results, and validating the framework across additional models and realistic workloads.

---

### Evidence notes

- **Primary public source:** [RobustAI GitHub development log](https://github.com/Reza-Moradi-m/physics-aware-testing) (accessed October 2026).
- **Project-author descriptions:** Additional local experiments, dashboards, neural-network comparisons, and broader failure-mode coverage as reported by Reza Moradi in project discussions and previous research résumé materials. These have not been independently re-executed for this draft.
- **Document type:** Working technical report; **not a peer-reviewed paper, accepted publication, or validated industrial reliability certification**.
