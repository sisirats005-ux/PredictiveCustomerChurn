# Concept Defense Cheat Sheet
*Quick-reference answers for a review or interview on this project. Each answer is written to show reasoning, not just recall a definition.*

**Quick index for the four core viva questions:**
- Why ROC-AUC over accuracy → Q2
- Why recall matters more than precision here → Q3
- How SHAP values are interpreted → Q1b
- Why Tuned XGBoost was the final model choice → Q7

---

### 1. Why does SHAP work?
SHAP is grounded in Shapley values from cooperative game theory. It treats each feature as a "player" contributing to the prediction (the "payout"), and fairly splits the difference between the model's average prediction and this specific prediction across all features. It does this by averaging each feature's marginal contribution across every possible order in which features could be "added" to the model. This gives a mathematically fair, consistent attribution — both direction (pushes risk up or down) and magnitude — for every individual prediction, not just a single global importance ranking.

### 1b. How do you actually read a SHAP plot? (the practical follow-up)
"Why SHAP works" (above) is the theory; this is what to say when someone points at a plot and asks "so what does this mean":
- **Beeswarm / summary plot** (one dot per customer per feature): position on the x-axis is the SHAP value — right of center pushes the prediction toward churn, left pushes toward retention. Color is the feature's own value (red = high, blue = low). So "red dots cluster on the right for `MonthlyCharges`" reads as: *high monthly charges push predicted churn risk up*. Features are ranked top-to-bottom by overall impact.
- **Bar plot**: the same ranking collapsed to a single number per feature — mean |SHAP value| — answering "which features matter most, on average, ignoring direction." Good for a one-slide executive summary; bad for explaining any single customer.
- **Waterfall plot** (one customer): starts at the model's average prediction (base value) and shows each feature pushing the prediction up or down, left to right, ending at that customer's actual predicted probability. This is the one to show when a retention agent asks "why did the model flag *this* customer specifically."
- **The common mistake to flag in a viva**: SHAP explains what the *model* learned, not ground truth about the world — a feature with a large SHAP value shows the model relied on it heavily, not that the feature causally drives churn (see also Q3 in the Limitations section on association vs. causation).

### 2. Why is ROC-AUC preferred over accuracy here?
Only ~26.6% of customers churn. A model that predicts "no churn" for everyone scores ~73% accuracy while catching zero real churners — accuracy rewards the model for ignoring the minority class entirely. ROC-AUC instead measures how well the model ranks positives above negatives across every possible decision threshold, so it can't be inflated just by favoring the majority class. It answers: "if I pick a random churner and a random non-churner, how often does the model correctly rank the churner as higher risk?"

### 3. Why does recall matter more than precision for churn?
The two error types have very different costs:
- **False negative** (missed churner): lose the customer, pay full re-acquisition cost.
- **False positive** (flagged non-churner): send a discount offer to someone who was staying anyway — a small, low-cost inefficiency.

Since missing a churner is far more expensive than a wasted retention offer, we optimize to catch as many true churners as possible — i.e., recall — even at some cost to precision.

### 4. Why use one-hot encoding instead of label/ordinal encoding?
Most categorical features here (`Contract`, `PaymentMethod`, `InternetService`) have no natural order. Encoding them as integers (0, 1, 2...) would falsely imply a ranking or distance relationship — e.g., that "Two year" contract is mathematically "more" than "Month-to-month" — which linear models and distance-based methods would misinterpret. One-hot encoding represents each category independently with no implied ordering.

### 5. Why use GridSearchCV?
Model performance is sensitive to hyperparameters (tree depth, learning rate, number of estimators), and the best combination isn't knowable in advance. GridSearchCV exhaustively evaluates every combination in a defined grid, scoring each via cross-validation, and returns the combination with the best average validation score — replacing manual guesswork with a systematic search.

### 6. Why is cross-validation needed?
A single train/test split can be lucky or unlucky depending on which customers happen to land in the test set — especially with a real-world, moderately-sized dataset like this one (7,043 rows). K-fold cross-validation rotates through multiple train/validation splits and averages the score, giving a more reliable estimate of how well the model generalizes. It's also what makes the GridSearchCV comparison between hyperparameter combinations fair — every combination is judged on the same multi-fold basis rather than one arbitrary split.

### 7. Why was Tuned XGBoost selected over the other four models?
Look at the actual comparison table (`outputs/metrics/model_comparison.csv`) — the choice isn't arbitrary, it follows directly from the metric we said mattered most (recall), with everything else backing it up:

| Model | Recall | ROC-AUC | Why it wasn't chosen |
| :--- | :---: | :---: | :--- |
| **Tuned XGBoost** | **0.8102** | 0.8442 | **Selected** — highest recall by a clear margin |
| Logistic Regression | 0.7834 | 0.8417 | 2.7 points lower recall; also assumes linear decision boundaries, which underfit the interaction effects (e.g. Contract × MonthlyCharges) that SHAP later shows matter |
| Tuned Random Forest | 0.7674 | 0.8423 | 4.3 points lower recall than tuned XGBoost, despite similar AUC |
| Baseline XGBoost | 0.6738 | 0.8220 | Untuned — shows why the GridSearchCV tuning step mattered (+13.6 recall points after tuning) |
| Baseline Random Forest | 0.5000 | 0.8243 | Highest precision (0.62) but worst recall — it plays it "safe" and misses half of all real churners, which is the opposite of what this business problem needs |

The pattern worth pointing out in a review: **AUC is close across all five models (0.82–0.85)** — the models mostly agree on *ranking* risk similarly. What separates them is **recall**, which depends on the tuned decision threshold and how well the model handles class imbalance (`scale_pos_weight` for XGBoost, `class_weight='balanced'` for RF/LR). Tuned XGBoost wins specifically because gradient boosting's sequential error-correction combined with the imbalance-aware tuning captured more true churners than the alternatives, without a proportionally large accuracy trade-off (0.741 vs. the baseline RF's 0.787 — only a 4.6-point accuracy cost for an 8-point recall gain, which is the right trade for this business problem).

---

### Bonus: anticipated pushback and how to handle it

**"Why isn't this deployed as a live app?"**
It is, as a demo: `streamlit run app.py` loads the serialized model and preprocessor and scores a single customer interactively (README §11). What's *not* there yet is a hardened production deployment — auth, batch scoring, request logging, containerization, drift monitoring — because the pipeline artifacts are already deployment-shaped (model and preprocessor are serialized separately in `models/`), so wrapping them in a FastAPI endpoint behind a proper CI/CD and monitoring setup is the next incremental step, not a redesign. (See README §14, Future MLOps Scope.)

**"How did you calculate the revenue impact numbers?"**
They're explicitly labeled as an illustrative scenario model (see `business_insight_report.md` §5) — the dataset doesn't include ConnectTel's real customer count or campaign costs, so the report states its assumptions (100K customers, $65 ARPU, $250 CAC, 25% campaign success rate) transparently rather than presenting a false-precision figure. The point is to show the shape of the argument — recall → customers saved → dollars retained — not to claim an exact number.

**"Is 0.845 AUC good?"**
It's solid, not exceptional — and that's fine, because the metric that actually drives business value here is recall (81%), not AUC in isolation. AUC confirms the model has genuine, stable ranking ability across thresholds; recall is what determines how many real churners the retention team actually gets a shot at saving.
