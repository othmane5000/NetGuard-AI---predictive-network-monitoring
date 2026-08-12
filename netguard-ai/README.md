# NetGuard AI — Network Failure Prediction & Predictive Monitoring System

Predictive, explainable network health monitoring built on supervised machine
learning (logistic regression, linear regression, gradient descent,
regularization) — not a black box, and not a simple online/offline checker.

> Given current and historical telemetry from routers, switches, firewalls,
> access points, and servers, NetGuard AI estimates the probability that a
> device is at risk of failure or serious degradation in the near future,
> and explains *why*.

---

## 1. Problem statement & motivation

Traditional network monitoring (SNMP polling, simple threshold alerts) tells
you a device is *already* down or *already* over a fixed CPU/temperature
threshold. It does not answer the more useful question: **"which of my 40
devices is quietly heading toward failure right now, before an outage
happens?"**

This project reframes network monitoring as a **supervised learning
problem**: a binary classification task (will this device experience a
failure/degradation event?) supported by a regression task (what will its
degradation score look like a few hours from now?), built end-to-end with
the methodology from *Supervised Machine Learning: Regression and
Classification* — cost functions, gradient descent, regularization,
train/test splitting, and proper evaluation — rather than opaque
`model.fit()` calls.

## 2. The exact ML problem

| | |
|---|---|
| **Task 1 (primary)** | Binary classification — will device *d* experience a `failure` event at time *t*, given its metrics at *t* and its recent history? |
| **Task 2 (secondary)** | Regression — what will device *d*'s composite **degradation score** (0–100) be **6 hours from now**, given its metrics at *t*? |
| **Target (classification)** | `failure` ∈ {0, 1} |
| **Target (regression)** | `degradation_score_future` ∈ [0, 100], a deterministic composite index (documented formula, §6) computed 6h ahead |
| **Unit of prediction** | One device, one point in time (hourly granularity) |
| **Positive class prevalence** | ~8.3% (realistic — most devices are healthy most of the time) |

### How failure labels are defined
In the synthetic generator (§4), a device's failure label at each hourly
timestep is drawn from a Bernoulli distribution whose probability is a
logistic function of that timestep's *z*-scored metrics (packet loss,
interface errors, temperature, CPU, latency, memory, bandwidth,
maintenance backlog, uptime) plus noise. This mirrors how a real "was there
an incident this hour" label would correlate with stress metrics, without
being a trivial rule (`if cpu > 90: failure = 1`) — the model has to learn
the relative importance of each factor, same as it would from a real
incident-ticket-linked dataset.

---

## 3. Dataset

### 3.1 Why synthetic data (explicitly stated decision)
Public datasets combining continuous device telemetry (CPU, memory,
temperature, packet loss, latency, interface errors) *with* labeled
hardware/network failure events are not available in the open domain —
vendors treat this as proprietary operational data. Datasets that *are*
public and network-related (NSL-KDD, CICIDS, Kaggle "network anomaly" sets)
address **security intrusion detection** (classifying attack traffic), a
different problem with a different feature schema and label semantics, and
would not honestly represent "predict hardware/performance failure from
health telemetry."

**Decision:** build a synthetic dataset with *causally grounded*
relationships between metrics and failure risk (documented in
`src/data_processing/generate_data.py`), rather than force-fitting an
intrusion-detection dataset onto a health-monitoring problem. This is a
standard, defensible approach in applied ML when no matching dataset
exists — as long as the generative logic is realistic and transparent,
which is the point of this section.

### 3.2 Schema

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Hourly reading |
| `device_id` | string | e.g. `SW-014`, `FW-003` |
| `device_type` | categorical | router / switch / firewall / access_point / server |
| `cpu_usage` | % | Processor utilization |
| `memory_usage` | % | RAM utilization |
| `temperature` | °C | Internal/ambient device temperature |
| `bandwidth_usage` | % | Link utilization vs capacity |
| `packet_loss` | % | Packets dropped in transit |
| `latency` | ms | Round-trip response time |
| `interface_errors` | count/hr | CRC/frame errors on interfaces |
| `active_connections` | count | Concurrent sessions/associations |
| `uptime_hours` | hours | Time since last reboot |
| `traffic_in` / `traffic_out` | Mbps | Inbound/outbound throughput |
| `previous_failures` | count | **Cumulative past failures for this device, shifted so the current row never counts itself** (leakage-safe) |
| `maintenance_days` | days | Days since last scheduled maintenance |
| `failure` | 0/1 | **Target.** Did a failure/degradation event occur at this timestep? |
| `failure_type` | categorical | `hardware_failure` / `connectivity_failure` / `performance_degradation` / `none` — **descriptive only, dropped before training (leaks the target)** |

40 simulated devices × 30 days hourly = 28,800 rows, 8.3% positive class.

### 3.3 Causal relationships built into the generator
- Packet loss ↑ and interface errors ↑ → failure risk ↑ (connectivity/hardware stress)
- Temperature ↑ (thermal episodes) → failure risk ↑, often co-moving with CPU throttling behavior
- CPU/memory overload episodes → failure risk ↑, latency ↑
- Network congestion episodes → bandwidth ↑, latency ↑, packet loss ↑ together (realistic co-movement, not independent spikes)
- Long uptime / maintenance backlog → small, realistic risk increase (not dominant)
- ~30% of devices go through a distinct degradation *episode* (ramping stress over 12–72 hours) before any failure — failures are not instantaneous, random events

### 3.4 Limitations of the synthetic data (stated explicitly)
- It encodes the **author's documented assumptions** about how metrics relate to failures, not measurements from real hardware. Coefficients used to validate the generator are not claimed to be universally true.
- The label-generating function is logistic, which naturally favors logistic regression at recovering it — this makes the model comparison meaningful for *methodology* but the absolute performance numbers below should not be read as "this is how well the model would work on your real network."
- No dataset field currently identifies a *specific physical component* — see §9 on why the app never claims to.

---

## 4. Feature engineering (and how leakage is avoided)

Beyond the 12 raw telemetry columns, `src/feature_engineering/build_features.py`
adds:

- **Rolling statistics** (6h window): moving average & std-dev of CPU, temperature, packet loss, latency, interface errors — uses the current + past 6 readings, which *is* legitimately available at prediction time.
- **Trend features**: current value minus the value 12 hours ago (`.shift(12)`), for CPU, packet loss, latency, interface errors, bandwidth — captures direction/speed of change using only past data.
- **Traffic growth %**: percentage change in traffic in/out over the trend window.
- **Spike/anomaly z-scores**: `(current - rolling_mean) / rolling_std`, with binary flags for CPU spikes and temperature anomalies.
- **Device age indicators**: uptime in days, long-uptime flag, recent-failure flag.
- **Composite ratios**: error rate (errors per connection), loss-over-latency.
- **One-hot encoded device type.**

**Explicitly excluded from the feature set:** `failure_type` (only exists
*because* a failure happened — pure leakage) and any raw generator-internal
probability. `device_id` and `timestamp` are identifiers, not model inputs.
`tests/test_feature_engineering.py` includes an automated check that
truncating a device's history after time *t* does not change its rolling
features at *t* — i.e., no feature secretly depends on future rows.

---

## 5. Models

### Model 1 — Logistic Regression (baseline)
Implemented **manually with NumPy** (sigmoid, binary cross-entropy cost,
batch gradient descent) and with **scikit-learn**, side by side, to
demonstrate the mechanics behind `sklearn.linear_model.LogisticRegression`
rather than treating it as a black box. See
`src/models/manual_implementations.py` for the from-scratch cost function
and gradient derivation (documented inline).

### Model 2 — Regularized Logistic Regression
Same manual + sklearn pairing, with an **L2 penalty** added to the cost
function to reduce overfitting, `C` (inverse of λ) tuned via 3-fold
`GridSearchCV` **optimizing for recall** (see §7 on why). The sklearn
version additionally uses `class_weight="balanced"` to counteract the 8.3%
class imbalance — the manual implementation does **not** yet support class
weighting, which is why its regularized result is close to its baseline
result (documented limitation, not a bug — see Model Comparison table).

### Model 3 — Regression: 6-hour-ahead degradation forecast
A composite **degradation score** (0–100) is defined as a documented,
weighted combination of normalized packet loss, interface errors,
temperature, CPU, and latency (see `compute_degradation_score()` in
`train_regression.py`). Predicting this score *at the same instant* from
the same raw inputs would be a trivial, circular regression (R²→1.0,
uninteresting). Instead, the model predicts the score **6 hours into the
future** using only present/past features — a genuine forecasting task.
Implemented manually (batch gradient descent, MSE cost, L2/ridge option)
and with `sklearn.linear_model.LinearRegression` / `Ridge`.

---

## 6. Evaluation

**Split strategy:** by **device**, not by row (80% of devices → train, 20%
→ test). Splitting by row would let the model see other timesteps from the
same device in both train and test — a subtle leakage risk in
per-entity time series that this project avoids deliberately.

### Classification results (test set: 8 held-out devices, 5,760 rows)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (manual NumPy) | 0.945 | 0.784 | 0.491 | 0.604 | 0.954 |
| Baseline (sklearn) | 0.947 | 0.769 | 0.552 | 0.642 | 0.959 |
| **Regularized (sklearn, class-balanced) — production model** | 0.901 | 0.459 | **0.871** | 0.601 | 0.959 |
| Regularized (manual, no class-weighting) | 0.945 | 0.784 | 0.491 | 0.604 | 0.954 |

**Why the production model is the regularized/balanced one despite lower
accuracy:** in network monitoring, a missed real failure (false negative)
is more costly than an unnecessary investigation (false positive) — see
§7. The balanced/regularized model catches **87% of real failures**
instead of 49–55%, at the cost of more false alarms (precision drops to
0.46). This tradeoff is deliberate and configurable via the classification
threshold, not an oversight.

### Regression results (6h-ahead degradation score forecast)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (manual) | 1.074 | 1.627 | 0.811 |
| Linear Regression (sklearn) | 1.073 | 1.626 | 0.811 |
| Ridge λ=5 (manual) | 1.074 | 1.627 | 0.811 |
| Ridge λ=5 (sklearn) | 1.073 | 1.626 | 0.811 |

Manual and sklearn implementations converge to essentially the same
solution in both tasks, which is the intended validation that the
from-scratch gradient descent implementation is correct.

### Precision/recall tradeoff, explained
- **False negative** (missed failure): device fails silently, outage/service impact, no warning given. **High cost.**
- **False positive** (false alarm): engineer spends a few minutes checking a device that turns out fine. **Low cost.**
Given this asymmetry, the project tunes toward recall, accepting a higher
false-positive rate. The classification threshold (default 0.5, used to
derive the confusion matrix above) and the risk-level bands (§7) are two
*independent, both-configurable* knobs for tuning this tradeoff further in
a real deployment.

---

## 7. Risk engine

Failure probability → risk level (default, configurable thresholds):

| Probability | Risk |
|---|---|
| 0 – 30% | LOW |
| 30 – 60% | MEDIUM |
| 60 – 80% | HIGH |
| 80 – 100% | CRITICAL |

**Limitation, stated explicitly:** these thresholds are a reasonable
starting point, not a scientifically optimal cutoff. A real deployment
should tune them against the organization's actual cost tolerance for
false alarms vs. missed failures, and potentially set different bands per
device criticality tier (a core switch and an access point don't carry the
same operational risk at the same failure probability).

**Important scope limitation:** NetGuard AI predicts *risk*, not the
specific failing component. The dataset has no component-level ground
truth, so the app never claims "the power supply will fail" — only
"packet loss, interface errors, and temperature are elevated," which is
what the data actually supports.

---

## 8. Explainability

Because the production classifier is a **linear model**, its own
standardized coefficients *are* the explanation — no separate black-box
explainability library (e.g. SHAP) is required for this version. For a
given device, `src/prediction/risk_engine.py::explain_prediction()`
computes each feature's contribution to the logit
(`coefficient × standardized_value`), ranks them, and surfaces the top
drivers with a severity tag (LOW/MEDIUM/HIGH/VERY HIGH relative to the
device's strongest driver) and a concrete recommended action, e.g.:

```
CORE-SWITCH-01 — Failure probability: 87% — Risk: CRITICAL
  Packet Loss         VERY HIGH   → Inspect physical links / switch ports
  Interface Errors    HIGH        → Inspect interfaces and cabling
  Temperature         HIGH        → Check cooling / airflow
  CPU Usage           MEDIUM      → Investigate high-CPU processes
```

---

## 9. Networking concepts this project relies on

| Concept | Relevance here |
|---|---|
| **Router / Switch / Firewall / Access Point / Server** | The five monitored device types, each with a different realistic operating envelope (e.g. firewalls run hotter/higher-CPU due to deep packet inspection; access points show more connection churn) |
| **Packet loss** | % of packets that never arrive — a primary connectivity-health signal |
| **Latency** | Round-trip delay; rises under congestion or routing problems |
| **Bandwidth / throughput** | Capacity vs. actual data moved; sustained high utilization → congestion |
| **Interface errors** | CRC/frame errors, usually cabling/hardware-level faults |
| **Congestion** | Too much traffic for available capacity → latency ↑, packet loss ↑ together (modeled as co-moving in the generator) |
| **CPU/memory impact** | Sustained overload can degrade forwarding performance and increase latency/loss |

**Failure-type taxonomy used (distinguished, not conflated):**
1. **Hardware failure** — component-level fault (elevated interface errors, thermal stress)
2. **Network connectivity failure** — loss of reachability (elevated packet loss)
3. **Performance degradation** — still reachable but slow/overloaded (elevated CPU, latency)
4. **Service failure** — out of scope for this dataset (would require application-layer/service-health signals not present here — noted as a Version 2 candidate)

---

## 10. Project architecture

```
netguard-ai/
├── data/
│   ├── raw/                  # generated synthetic telemetry
│   └── processed/            # engineered feature table
├── notebooks/                # optional exploratory analysis
├── src/
│   ├── data_processing/      # generate_data.py
│   ├── feature_engineering/  # build_features.py
│   ├── models/                # manual_implementations.py, train_classification.py, train_regression.py
│   ├── evaluation/            # metrics.py, generate_reports.py
│   └── prediction/            # risk_engine.py, predict.py
├── app/
│   └── streamlit_app.py      # NOC dashboard
├── models/                    # trained model artifacts (.joblib)
├── reports/                   # generated plots + metrics json/csv
├── tests/                     # pytest unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

Data processing, feature engineering, model training, prediction, and the
UI are fully separated — the Streamlit app only ever *loads* trained
artifacts and calls the prediction pipeline; it never trains models
itself.

---

## 11. Installation & running it

```bash
git clone <your-repo-url>
cd netguard-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python3 src/data_processing/generate_data.py

# 2. Build the engineered feature table (optional standalone check)
python3 src/feature_engineering/build_features.py

# 3. Train the classification models (Model 1 + Model 2)
python3 src/models/train_classification.py

# 4. Train the regression model (Model 3)
python3 src/models/train_regression.py

# 5. Generate static report plots (confusion matrix, ROC, feature importance)
python3 src/evaluation/generate_reports.py

# 6. Run the tests
pytest tests/ -v

# 7. Launch the dashboard
streamlit run app/streamlit_app.py
```

**Expected output after step 3-4:** printed accuracy/precision/recall/F1/
ROC-AUC for each classification variant, and MAE/RMSE/R² for each
regression variant (matching the tables in §6), plus saved model artifacts
in `models/`.

**Expected output after step 7:** a browser tab opens at
`http://localhost:8501` showing the NOC dashboard with 40 simulated
devices, live-computed risk scores, and four navigable sections (Dashboard,
Device Monitoring, Device Details, Analytics).

---

## 12. Dashboard overview

- **Dashboard** — fleet KPIs (total/healthy/warning/critical/predicted failures), top-10 highest-risk devices, risk distribution donut.
- **Device Monitoring** — filterable/searchable table of all devices with live metrics and risk level.
- **Device Details** — per-device current metrics, "why is this device at risk" explanation with recommended actions, and historical trend charts (CPU/memory/temp, packet loss/latency, interface errors).
- **Analytics** — failure-type and risk distribution, model comparison table, confusion matrix, ROC curve, feature-importance chart, fleet-wide historical trends.

---

## 13. Results summary (for interview discussion)

- Recall-optimized production classifier catches **87%** of simulated
  failures (vs. 49–55% for the unweighted baseline), at a deliberately
  accepted precision cost — a defensible operational tradeoff, not an
  accident.
- Manual NumPy implementations of logistic and linear regression converge
  to metrics within noise of scikit-learn's equivalents, validating the
  from-scratch gradient descent/cost-function code.
- The 6-hour-ahead degradation forecast achieves R² ≈ 0.81 — meaningfully
  predictive without being suspiciously perfect (the earlier same-instant
  formulation, kept out of the final version, hit R² ≈ 1.0 because it was
  trivially circular — documented here rather than hidden).
- Zero data leakage confirmed by an automated test that truncates a
  device's history and checks rolling features don't change.

## 14. Limitations (stated explicitly, not hidden)

- Trained and validated on **synthetic** data with documented, author-defined causal assumptions — absolute metric values should not be read as "this is what accuracy would be on a real production network."
- The system predicts **risk of failure/degradation**, not the exact physical component that will fail — the dataset provides no component-level ground truth.
- Risk-level thresholds (§7) are a reasonable default, not a tuned optimum for any specific organization.
- The regression task forecasts a documented composite index, not a directly-measured real-world label (no such label exists in the raw telemetry).
- Manual logistic regression does not currently implement class-weighting, so its regularized variant does not show the same recall improvement as sklearn's balanced version (documented in §6, not silently glossed over).

## 15. Version 2 (proposed, not implemented here)

- Real telemetry via SNMP / syslog / vendor device APIs
- Prometheus + Grafana integration for live metrics ingestion
- True real-time streaming inference + alerting (email/Slack/PagerDuty)
- Time-series-native models (e.g. LSTM/temporal CNN) for multi-step forecasting, compared against this project's linear baselines
- Unsupervised anomaly detection layer (e.g. isolation forest) as a complement to supervised failure classification
- SHAP-based explainability once/if a non-linear model is introduced
- Dockerized deployment + REST API (FastAPI) in front of the prediction pipeline
- Cloud deployment (e.g. AWS/GCP) with a managed feature store

---

## Tech stack

Python · NumPy · Pandas · Scikit-learn · Matplotlib · Streamlit · Joblib · Pytest
