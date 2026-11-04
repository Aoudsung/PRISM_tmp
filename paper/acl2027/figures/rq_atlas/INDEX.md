# RQ1–RQ5 完整图集

按认可样张统一绘制；图内不放解释性页脚。

| PDF 页码 | 图表 | 图件 |
|---|---|---|
| 1 | RQ1 · Test utility and success (%) | [PDF](plots/rq1_01_test_results_full.pdf) · [PNG](plots/rq1_01_test_results_full.png) |
| 2 | RQ1 · Focused utility gains | [PDF](plots/rq1_03_utility_gain_overview.pdf) · [PNG](plots/rq1_03_utility_gain_overview.png) |
| 3 | RQ1 · Individual seeds: Qwen2.5-7B | [PDF](plots/rq1_04_seed_variation_qwen25_7b.pdf) · [PNG](plots/rq1_04_seed_variation_qwen25_7b.png) |
| 4 | RQ1 · Individual seeds: Llama-3.1-8B | [PDF](plots/rq1_04_seed_variation_llama31_8b.pdf) · [PNG](plots/rq1_04_seed_variation_llama31_8b.png) |
| 5 | RQ1 · Individual seeds: Qwen3.8-27B | [PDF](plots/rq1_04_seed_variation_qwen38_27b.pdf) · [PNG](plots/rq1_04_seed_variation_qwen38_27b.png) |
| 6 | RQ2 · Mechanism-control gains | [PDF](plots/rq2_01_mechanism_controls.pdf) · [PNG](plots/rq2_01_mechanism_controls.png) |
| 7 | RQ2 · Aligned weighting versus Unweighted | [PDF](plots/rq2_02_aligned_weighting.pdf) · [PNG](plots/rq2_02_aligned_weighting.png) |
| 8 | RQ2 · Utility and interaction cost | [PDF](plots/rq2_03_cost_utility.pdf) · [PNG](plots/rq2_03_cost_utility.png) |
| 9 | RQ3  ·  Sampling coverage and gradient agreement | [PDF](plots/rq3_01_coverage_and_gradients.pdf) · [PNG](plots/rq3_01_coverage_and_gradients.png) |
| 10 | RQ3  ·  Exact and Monte Carlo variance–cost comparison | [PDF](plots/rq3_02_variance_cost.pdf) · [PNG](plots/rq3_02_variance_cost.png) |
| 11 | RQ4 · AppWorld utility evolution | [PDF](plots/rq4_01_stage_evolution_appworld.pdf) · [PNG](plots/rq4_01_stage_evolution_appworld.png) |
| 12 | RQ4 · PlanBench-XL utility evolution | [PDF](plots/rq4_01_stage_evolution_planbench_xl.pdf) · [PNG](plots/rq4_01_stage_evolution_planbench_xl.png) |
| 13 | RQ4 · AppWorld complete stage values (Stages 0–3) | [PDF](plots/rq4_01_stage_values_appworld.pdf) · [PNG](plots/rq4_01_stage_values_appworld.png) |
| 14 | RQ4 · PlanBench-XL complete stage values (Stages 0–3) | [PDF](plots/rq4_01_stage_values_planbench_xl.pdf) · [PNG](plots/rq4_01_stage_values_planbench_xl.png) |
| 15 | RQ4 · AppWorld feedback | [PDF](plots/rq4_02_feedback_utilization_appworld.pdf) · [PNG](plots/rq4_02_feedback_utilization_appworld.png) |
| 16 | RQ4 · PlanBench-XL feedback | [PDF](plots/rq4_02_feedback_utilization_planbench_xl.pdf) · [PNG](plots/rq4_02_feedback_utilization_planbench_xl.png) |
| 17 | RQ4 · Feedback stage endpoints | [PDF](plots/rq4_02_feedback_stage_change.pdf) · [PNG](plots/rq4_02_feedback_stage_change.png) |
| 18 | RQ4 · AppWorld trajectory outcomes | [PDF](plots/rq4_03_trajectory_outcomes_appworld.pdf) · [PNG](plots/rq4_03_trajectory_outcomes_appworld.png) |
| 19 | RQ4 · PlanBench-XL trajectory outcomes | [PDF](plots/rq4_03_trajectory_outcomes_planbench_xl.pdf) · [PNG](plots/rq4_03_trajectory_outcomes_planbench_xl.png) |
| 20 | RQ4 · Complete feedback utilization and conditional recovery | [PDF](plots/rq4_04_feedback_and_recovery_values.pdf) · [PNG](plots/rq4_04_feedback_and_recovery_values.png) |
| 21 | RQ5 · Budget response: focused comparison | [PDF](plots/rq5_01_budget_response.pdf) · [PNG](plots/rq5_01_budget_response.png) |
| 22 | RQ5 · Complete budget utility and success values | [PDF](plots/rq5_02_budget_values_full.pdf) · [PNG](plots/rq5_02_budget_values_full.png) |
| 23 | RQ5 · Training split location | [PDF](plots/rq5_03_split_schedule.pdf) · [PNG](plots/rq5_03_split_schedule.png) |
| 24 | Training · Batch returns | [PDF](plots/supp_01_training_returns.pdf) · [PNG](plots/supp_01_training_returns.png) |
| 25 | Training · AppWorld losses across three models | [PDF](plots/supp_02_appworld_losses_combined.pdf) · [PNG](plots/supp_02_appworld_losses_combined.png) |
| 26 | Training · PlanBench-XL losses across three models | [PDF](plots/supp_03_planbench_losses_combined.pdf) · [PNG](plots/supp_03_planbench_losses_combined.png) |
| 27 | Resources · AppWorld · Complete training work and timing ledger | [PDF](plots/supp_04_resources_appworld.pdf) · [PNG](plots/supp_04_resources_appworld.png) |
| 28 | Resources · PlanBench-XL · Complete training work and timing ledger | [PDF](plots/supp_05_resources_planbench_xl.pdf) · [PNG](plots/supp_05_resources_planbench_xl.png) |
| 29 | Resources · Update and snapshot counts | [PDF](plots/supp_04_resource_work_counts.pdf) · [PNG](plots/supp_04_resource_work_counts.png) |

## 数据口径

<details><summary>1. RQ1 | Test utility and success (%)</summary>

Complete 14-method table with three-seed means and sample SD for both test utility and success rate. Model blocks each contain AppWorld and PlanBench-XL; best means are bold and second-best underlined.

来源：metrics/main_results

</details>

<details><summary>2. RQ1 | Focused utility gains</summary>

Mean-based utility differences multiplied by 100. The strongest named peer is selected within each saved model/benchmark setting; its name is displayed.

来源：comparisons/method_comparisons

</details>

<details><summary>3. RQ1 | Individual seeds: Qwen2.5-7B</summary>

All 14 methods and three seeds; each horizontal interval is the stored sample SD around the mean. Raw seed points use fixed marker shapes.

来源：metrics/seed_results, metrics/main_results

</details>

<details><summary>4. RQ1 | Individual seeds: Llama-3.1-8B</summary>

All 14 methods and three seeds; each horizontal interval is the stored sample SD around the mean. Raw seed points use fixed marker shapes.

来源：metrics/seed_results, metrics/main_results

</details>

<details><summary>5. RQ1 | Individual seeds: Qwen3.8-27B</summary>

All 14 methods and three seeds; each horizontal interval is the stored sample SD around the mean. Raw seed points use fixed marker shapes.

来源：metrics/seed_results, metrics/main_results

</details>

<details><summary>6. RQ2 | Mechanism-control gains</summary>

The first five rows compare return/sampling controls; the final two rows compare weighting/preference controls. Values are mean utility differences ×100.

来源：metrics/main_results

</details>

<details><summary>7. RQ2 | Aligned weighting versus Unweighted</summary>

Mean-based contrasts. Relative cost reduction uses Unweighted as denominator; feedback values use the fixed 24-task development diagnostic.

来源：comparisons/method_comparisons/reference_role=unweighted

</details>

<details><summary>8. RQ2 | Utility and interaction cost</summary>

Each point is one method's three-seed mean with SD error crosshairs on focused methods; dashed arrows show the Unweighted-to-SPAR shift.

来源：metrics/main_results

</details>

<details><summary>9. RQ3  |  Sampling coverage and gradient agreement</summary>

Exact two-stage setting. MC: 10,000 episodes. Gradient coordinates follow the stored parameter-vector order.

来源：metrics/exact_verification

</details>

<details><summary>10. RQ3  |  Exact and Monte Carlo variance–cost comparison</summary>

Percentages compare stratified with RLOO (Exact / Monte Carlo); MC uses 10,000 episodes.

来源：metrics/exact_verification

</details>

<details><summary>11. RQ4 | AppWorld utility evolution</summary>

Three-seed means; SPAR and Unweighted bands are sample SD. All methods also appear in complete stage-value tables.

来源：metrics.stage_results

</details>

<details><summary>12. RQ4 | PlanBench-XL utility evolution</summary>

Three-seed means; SPAR and Unweighted bands are sample SD. All methods also appear in complete stage-value tables.

来源：metrics.stage_results

</details>

<details><summary>13. RQ4 | AppWorld complete stage values (Stages 0–3)</summary>

Mean +/- sample SD over three seeds across all four training stages (0–3). Column maxima are bold where values differ.

来源：metrics.stage_results

</details>

<details><summary>14. RQ4 | PlanBench-XL complete stage values (Stages 0–3)</summary>

Mean +/- sample SD over three seeds across all four training stages (0–3). Column maxima are bold where values differ.

来源：metrics.stage_results

</details>

<details><summary>15. RQ4 | AppWorld feedback</summary>

Fixed 24-task diagnostic; points are means and whiskers sample SD over three seeds.

来源：metrics.main_results.final_dev.delta_use

</details>

<details><summary>16. RQ4 | PlanBench-XL feedback</summary>

Fixed 24-task diagnostic; points are means and whiskers sample SD over three seeds.

来源：metrics.main_results.final_dev.delta_use

</details>

<details><summary>17. RQ4 | Feedback stage endpoints</summary>

Fixed 24-task diagnostic; three-seed mean endpoints.

来源：comparisons.stage_changes

</details>

<details><summary>18. RQ4 | AppWorld trajectory outcomes</summary>

Three-seed pooled counts; successful evaluations without a recorded failure, recovered successes, and final failures.

来源：comparisons.trajectory_outcomes

</details>

<details><summary>19. RQ4 | PlanBench-XL trajectory outcomes</summary>

Three-seed pooled counts; successful evaluations without a recorded failure, recovered successes, and final failures.

来源：comparisons.trajectory_outcomes

</details>

<details><summary>20. RQ4 | Complete feedback utilization and conditional recovery</summary>

Panel (a): fixed 24-task diagnostic mean +/- sample SD over three seeds. Panel (b): recovered successes / evaluations with recorded failure and pooled rates. Column maxima are bold.

来源：metrics.main_results.final_dev.delta_use, comparisons.trajectory_outcomes

</details>

<details><summary>21. RQ5 | Budget response: focused comparison</summary>

AppWorld means; the vertical line is default budget. Tight-budget labels are SPAR minus Unweighted utility x100.

来源：metrics.budgets

</details>

<details><summary>22. RQ5 | Complete budget utility and success values</summary>

All 14 methods and three budgets on AppWorld; three-seed means for both utility and success rate (%). Highest values per column are bold.

来源：metrics.budgets

</details>

<details><summary>23. RQ5 | Training split location</summary>

AppWorld; point = mean, whisker = sample SD over three seeds. Uniform schedules sample T from the displayed set.

来源：metrics.split_steps

</details>

<details><summary>24. Training | Batch returns</summary>

All 14 methods retained. Lines are seed means; bands are sample SD across seeds 42, 43, 44.

来源：training_progress

</details>

<details><summary>25. Training | AppWorld losses across three models</summary>

One panel per nonzero-loss method; each panel retains the original loss units and a shared scale for its three models. Points are three-seed means; thin intervals are sample SD. Models use fixed line styles and circle/square/diamond markers; color retains method identity. Stages have horizontal offsets -0.045/0/+0.045 solely to separate markers and intervals. The final cell preserves methods whose recorded losses are zero for all models, stages and seeds.

来源：training_progress

</details>

<details><summary>26. Training | PlanBench-XL losses across three models</summary>

One panel per nonzero-loss method; each panel retains the original loss units and a shared scale for its three models. Points are three-seed means; thin intervals are sample SD. Models use fixed line styles and circle/square/diamond markers; color retains method identity. Stages have horizontal offsets -0.045/0/+0.045 solely to separate markers and intervals. The final cell preserves methods whose recorded losses are zero for all models, stages and seeds.

来源：training_progress

</details>

<details><summary>27. Resources | AppWorld | Complete training work and timing ledger</summary>

Per-seed training totals averaged over three seeds. Panel (a) reports interaction steps, tokens, and wall-clock time; Panel (b) reports amortized model encoding, gradient update, and snapshot restore time.

来源：cost_ledgers, backbone_metadata

</details>

<details><summary>28. Resources | PlanBench-XL | Complete training work and timing ledger</summary>

Per-seed training totals averaged over three seeds. Panel (a) reports interaction steps, tokens, and wall-clock time; Panel (b) reports amortized model encoding, gradient update, and snapshot restore time.

来源：cost_ledgers, backbone_metadata

</details>

<details><summary>29. Resources | Update and snapshot counts</summary>

Counts are identical across all six model/benchmark settings. Snapshot saves equal restores.

来源：cost_ledgers

</details>

