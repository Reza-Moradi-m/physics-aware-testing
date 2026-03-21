# RobustAI Engine Audit Report

**Dataset:** nasa_fd001
**Profile:** aviation_extreme
**Profile Description:** High-altitude jet engine telemetry with tight latency deadlines and safety-critical monitoring.
**Safety Threshold:** 0.70
**Best Clean Accuracy:** 0.9506
**Worst Effective Accuracy:** 0.8389
**Worst Case:** model=LogisticRegression, constraint=staleness

## PASS

Safety gate did not trigger for this run.

## Result Summary

| model              | profile          | constraint           |   clean_accuracy |   effective_accuracy |        delta |
|:-------------------|:-----------------|:---------------------|-----------------:|---------------------:|-------------:|
| LogisticRegression | aviation_extreme | clean                |         0.950562 |             0.950562 |  0           |
| LogisticRegression | aviation_extreme | noise                |         0.950562 |             0.950756 | -0.000193874 |
| LogisticRegression | aviation_extreme | noise                |         0.950562 |             0.949787 |  0.000775494 |
| LogisticRegression | aviation_extreme | noise                |         0.950562 |             0.949399 |  0.00116324  |
| LogisticRegression | aviation_extreme | noise                |         0.950562 |             0.948236 |  0.00232648  |
| LogisticRegression | aviation_extreme | latency              |         0.950562 |             0.945909 |  0.00465297  |
| LogisticRegression | aviation_extreme | staleness            |         0.950562 |             0.94843  |  0.00213261  |
| LogisticRegression | aviation_extreme | bias_drift           |         0.950562 |             0.949593 |  0.000969368 |
| LogisticRegression | aviation_extreme | latency              |         0.950562 |             0.849748 |  0.100814    |
| LogisticRegression | aviation_extreme | staleness            |         0.950562 |             0.94746  |  0.00310198  |
| LogisticRegression | aviation_extreme | bias_drift           |         0.950562 |             0.947073 |  0.00348972  |
| LogisticRegression | aviation_extreme | latency              |         0.950562 |             0.849748 |  0.100814    |
| LogisticRegression | aviation_extreme | staleness            |         0.950562 |             0.940093 |  0.0104692   |
| LogisticRegression | aviation_extreme | bias_drift           |         0.950562 |             0.944746 |  0.00581621  |
| LogisticRegression | aviation_extreme | latency              |         0.950562 |             0.849748 |  0.100814    |
| LogisticRegression | aviation_extreme | staleness            |         0.950562 |             0.912369 |  0.0381931   |
| LogisticRegression | aviation_extreme | bias_drift           |         0.950562 |             0.934858 |  0.0157038   |
| LogisticRegression | aviation_extreme | latency              |         0.950562 |             0.849748 |  0.100814    |
| LogisticRegression | aviation_extreme | staleness            |         0.950562 |             0.838891 |  0.111671    |
| LogisticRegression | aviation_extreme | bias_drift           |         0.950562 |             0.900543 |  0.0500194   |
| LogisticRegression | aviation_extreme | intermittent_dropout |         0.950562 |             0.944746 |  0.00581621  |
| LogisticRegression | aviation_extreme | stuck_at_value       |         0.950562 |             0.939705 |  0.0108569   |
| LogisticRegression | aviation_extreme | saturation           |         0.950562 |             0.951144 | -0.000581621 |
| LogisticRegression | aviation_extreme | quantization         |         0.950562 |             0.94746  |  0.00310198  |
| LogisticRegression | aviation_extreme | packet_burst_loss    |         0.950562 |             0.945522 |  0.00504071  |
| MLP                | aviation_extreme | clean                |         0.949787 |             0.949787 |  0           |
| MLP                | aviation_extreme | noise                |         0.949787 |             0.949205 |  0.000581621 |
| MLP                | aviation_extreme | noise                |         0.949787 |             0.948817 |  0.000969368 |
| MLP                | aviation_extreme | noise                |         0.949787 |             0.94843  |  0.00135712  |
| MLP                | aviation_extreme | noise                |         0.949787 |             0.949399 |  0.000387747 |
| MLP                | aviation_extreme | latency              |         0.949787 |             0.94494  |  0.00484684  |
| MLP                | aviation_extreme | staleness            |         0.949787 |             0.950174 | -0.000387747 |
| MLP                | aviation_extreme | bias_drift           |         0.949787 |             0.947848 |  0.00193874  |
| MLP                | aviation_extreme | latency              |         0.949787 |             0.849748 |  0.100039    |
| MLP                | aviation_extreme | staleness            |         0.949787 |             0.949399 |  0.000387747 |
| MLP                | aviation_extreme | bias_drift           |         0.949787 |             0.946103 |  0.0036836   |
| MLP                | aviation_extreme | latency              |         0.949787 |             0.849748 |  0.100039    |
| MLP                | aviation_extreme | staleness            |         0.949787 |             0.94145  |  0.00833656  |
| MLP                | aviation_extreme | bias_drift           |         0.949787 |             0.941256 |  0.00853044  |
| MLP                | aviation_extreme | latency              |         0.949787 |             0.849748 |  0.100039    |
| MLP                | aviation_extreme | staleness            |         0.949787 |             0.913726 |  0.0360605   |
| MLP                | aviation_extreme | bias_drift           |         0.949787 |             0.931369 |  0.018418    |
| MLP                | aviation_extreme | latency              |         0.949787 |             0.849748 |  0.100039    |
| MLP                | aviation_extreme | staleness            |         0.949787 |             0.869911 |  0.0798759   |
| MLP                | aviation_extreme | bias_drift           |         0.949787 |             0.890461 |  0.0593253   |
| MLP                | aviation_extreme | intermittent_dropout |         0.949787 |             0.944552 |  0.00523459  |
| MLP                | aviation_extreme | stuck_at_value       |         0.949787 |             0.938736 |  0.0110508   |
| MLP                | aviation_extreme | saturation           |         0.949787 |             0.949981 | -0.000193874 |
| MLP                | aviation_extreme | quantization         |         0.949787 |             0.947073 |  0.00271423  |
| MLP                | aviation_extreme | packet_burst_loss    |         0.949787 |             0.944746 |  0.00504071  |
