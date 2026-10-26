# Second Look charts

Made by `make media` (scripts/media.py) from eval/results. Each image is 1600 px wide.

- `chart-1-summary.png`: "Same as last year" gave 758 held-out adults the truly cheapest 2027 plan 78% of the time, against 82% for the lowest premium. Average overpayment: $149 against $106. Source: `eval/results/step4_decision.json`.
- `chart-2-plan-choice.png`: Every plan-choosing strategy on the three real 2027 menus: share given the truly cheapest plan, mean and 90th-percentile overpayment, with 95% intervals. Source: `eval/results/step4_decision.json`.
- `chart-3-learning-curve.png`: Pinball loss and 10-90% band coverage against training-set size for three spend models. TabPFN v2 is within 5% of its full-data median loss at 200 people. Source: `eval/results/step2b_learning_curve.json`.
- `chart-4-plan-reader.png`: Three Gemma 4 builds on 260 key values from 20 held-out plan documents. gemma4:e4b-it-qat read 91.9% right, 3.5% wrong, and flagged 4.6% for checking. Source: `eval/results/step3_reader.json`.
