\# 🏦 Explainable Credit Risk Engine



A machine learning pipeline to predict credit default risk with full explainability. Built to demonstrate how AI can support transparent and auditable lending decisions in regional banking.



\---



\## 🚀 Live Demo

\[Launch App](https://your-app.streamlit.app) ← \*(will be updated after deployment)\*



\---



\## 📊 Models \& Results



| Model | AUC | Class 1 Recall | Class 1 F1 |

|---|---|---|---|

| Logistic Regression | 0.712 | 0.04 | 0.08 |

| Random Forest | 0.841 | 0.19 | 0.28 |

| \*\*XGBoost\*\* | \*\*0.869\*\* | \*\*0.20\*\* | \*\*0.29\*\* |



\---



\## 💰 Financial Impact Analysis



By optimizing the decision threshold from 0.50 to 0.24:



| Metric | Default (0.50) | Optimal (0.24) |

|---|---|---|

| Missed defaulters | 1,612 | 1,102 |

| Default losses | $14,508,000 | $9,918,000 |

| Total financial cost | $14,657,400 | $10,530,450 |



\## 📈 Key Visualizations



\### SHAP Summary — XGBoost

!\[SHAP Summary](images/shap\_summary\_xgb.png)



\### Threshold Analysis — Precision / Recall / F1

!\[Threshold Analysis](images/threshold\_analysis.png)



\### ROC Curve Comparison

!\[ROC Curve](images/roc\_curve\_updated.png)

