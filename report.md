# Predictive Maintenance System: Model Selection and Classification Approaches

## Abstract

This report presents a comprehensive predictive maintenance system developed using machine learning techniques applied to the AI4I 2020 Predictive Maintenance Dataset. The system implements three distinct predictive models: a multi-class failure type classifier, a binary health status classifier with continuous health scoring, and a regression model for Remaining Useful Life (RUL) estimation. This document provides detailed justification for model selection, classification approaches, feature engineering methodologies, and evaluation metrics employed in the development of this system.

---

## 1. Introduction

### 1.1 Problem Statement

Predictive maintenance represents a paradigm shift from traditional reactive and preventive maintenance strategies. Rather than responding to equipment failures after they occur or performing maintenance at fixed intervals regardless of equipment condition, predictive maintenance leverages sensor data and machine learning algorithms to anticipate failures before they happen.

The primary objectives of this predictive maintenance system are:

1. **Failure Type Prediction**: Classify the specific type of failure likely to occur
2. **Health Assessment**: Determine whether equipment is healthy or showing signs of degradation
3. **Remaining Useful Life Estimation**: Predict the operational time remaining before maintenance is required

### 1.2 Dataset Description

The AI4I 2020 Predictive Maintenance Dataset comprises 10,000 observations of machine operational parameters collected from industrial equipment. The dataset includes the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| Air temperature | Ambient air temperature | Kelvin (K) |
| Process temperature | Operating temperature | Kelvin (K) |
| Rotational speed | Spindle rotation speed | RPM |
| Torque | Applied torque | Newton-meters (Nm) |
| Tool wear | Cumulative tool usage | Minutes |
| Product Type | Quality category | H (High), M (Medium), L (Low) |

The dataset includes five distinct failure modes:
- **Tool Wear Failure (TWF)**: 46 instances (0.46%)
- **Heat Dissipation Failure (HDF)**: 115 instances (1.15%)
- **Power Failure (PWF)**: 95 instances (0.95%)
- **Overstrain Failure (OSF)**: 98 instances (0.98%)
- **Random Failure (RNF)**: 19 instances (0.19%)

The significant class imbalance, with 96.61% of observations representing normal operation, presents a substantial challenge for classification algorithms.

---

## 2. Feature Engineering

### 2.1 Physics-Based Feature Construction

A critical contribution of this work is the development of physics-based engineered features derived from domain knowledge of mechanical systems. Rather than relying solely on raw sensor measurements, we constructed features that capture the underlying physical phenomena associated with each failure mode.

#### 2.1.1 Temperature Differential

$$\Delta T = T_{process} - T_{air}$$

The temperature differential represents the heat dissipation capability of the system. Insufficient heat dissipation (low ΔT) combined with low rotational speed indicates potential Heat Dissipation Failure.

#### 2.1.2 Mechanical Power

$$P = \tau \cdot \omega = \tau \cdot \frac{2\pi \cdot RPM}{60}$$

Where τ represents torque and ω represents angular velocity. Power values outside the normal operating range (3,500W - 9,000W) indicate potential Power Failure conditions.

#### 2.1.3 Mechanical Strain

$$\sigma = W_{tool} \times \tau$$

The product of tool wear and torque represents the cumulative mechanical stress on the system. Strain values exceeding 11,000 indicate Overstrain Failure risk.

### 2.2 Binary Risk Indicators

Based on the physics-based features, binary risk indicators were constructed to capture threshold-based failure conditions:

| Indicator | Condition | Associated Failure |
|-----------|-----------|-------------------|
| HDF Risk | ΔT < 8.6K AND RPM < 1,380 | Heat Dissipation Failure |
| PWF Risk | P < 3,500W OR P > 9,000W | Power Failure |
| OSF Risk | σ > 11,000 | Overstrain Failure |
| TWF Risk | Tool wear ≥ 200 min | Tool Wear Failure |

### 2.3 Justification for Feature Engineering

The decision to employ physics-based feature engineering rather than relying solely on raw features is justified by several factors:

1. **Domain Knowledge Integration**: Incorporating known failure physics improves model interpretability and reliability
2. **Dimensionality Reduction**: Complex relationships are captured in meaningful derived features
3. **Improved Generalization**: Physics-based features generalize better to unseen operating conditions
4. **Feature Importance**: Experimental results confirm that engineered features (risk indicators) dominate feature importance rankings

---

## 3. Classification Approaches

This system employs three distinct classification paradigms, each selected to address specific predictive maintenance requirements.

### 3.1 Multi-Class Classification: Failure Type Prediction

#### 3.1.1 Problem Formulation

The failure type prediction task is formulated as a multi-class classification problem with six mutually exclusive classes:

$$y \in \{C_0, C_1, C_2, C_3, C_4, C_5\}$$

Where:
- $C_0$: No Failure
- $C_1$: Tool Wear Failure
- $C_2$: Heat Dissipation Failure
- $C_3$: Power Failure
- $C_4$: Overstrain Failure
- $C_5$: Random Failure

#### 3.1.2 Model Selection: Random Forest Classifier

The Random Forest algorithm was selected for multi-class failure prediction based on the following considerations:

**Theoretical Foundation**

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification. The algorithm employs two key techniques:

1. **Bagging (Bootstrap Aggregating)**: Each tree is trained on a bootstrap sample of the training data
2. **Feature Randomization**: At each split, only a random subset of features is considered

**Justification for Selection**

| Criterion | Random Forest Advantage |
|-----------|------------------------|
| Class Imbalance | Native support for class weighting via `class_weight="balanced"` |
| Non-linearity | Captures complex decision boundaries without explicit specification |
| Feature Interactions | Automatically models interactions between sensor readings |
| Robustness | Resistant to overfitting through ensemble averaging |
| Interpretability | Provides feature importance rankings |
| Scalability | Parallelizable training via `n_jobs=-1` |

**Alternative Algorithms Considered**

| Algorithm | Reason for Rejection |
|-----------|---------------------|
| Logistic Regression | Linear decision boundaries insufficient for complex failure patterns |
| Support Vector Machines | Computationally expensive for multi-class; requires careful kernel selection |
| Neural Networks | Requires larger datasets; less interpretable; longer training time |
| Naive Bayes | Independence assumption violated by correlated sensor readings |

#### 3.1.3 Handling Class Imbalance: SMOTE

The Synthetic Minority Over-sampling Technique (SMOTE) was employed to address the severe class imbalance in the dataset.

**SMOTE Algorithm**

For each minority class sample $x_i$:
1. Identify k-nearest neighbors in feature space
2. Select one neighbor $x_{nn}$ randomly
3. Generate synthetic sample: $x_{new} = x_i + \lambda(x_{nn} - x_i)$, where $\lambda \in [0,1]$

**Implementation Parameters**
- `k_neighbors=3`: Reduced from default due to small minority class sizes
- Resampling strategy: All classes balanced to majority class size

**Results**

| Dataset | Samples |
|---------|---------|
| Original Training Set | 8,000 |
| Resampled Training Set | 46,332 |

Each class was balanced to 7,722 samples, enabling the model to learn meaningful patterns from minority failure classes.

### 3.2 Binary Classification: Health Status Assessment

#### 3.2.1 Problem Formulation

The health classification task extends beyond simple failure prediction to identify machines showing early warning signs of degradation:

$$y \in \{Healthy, Unhealthy\}$$

A machine is classified as **Unhealthy** if:
- An actual failure occurred, OR
- Any risk indicator threshold is exceeded (proactive detection)

This formulation enables **proactive maintenance** by identifying at-risk equipment before failure occurs.

#### 3.2.2 Model Selection: Gradient Boosting Classifier

Gradient Boosting was selected for binary health classification based on:

**Theoretical Foundation**

Gradient Boosting builds an ensemble of weak learners (decision trees) sequentially, with each subsequent tree trained to correct the errors of the previous ensemble:

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

Where $h_m(x)$ is the weak learner fitted to the negative gradient of the loss function.

**Justification for Selection**

| Criterion | Gradient Boosting Advantage |
|-----------|----------------------------|
| Accuracy | Generally achieves higher accuracy than Random Forest for binary tasks |
| Probability Calibration | Well-calibrated probability estimates for health scoring |
| Sequential Learning | Focuses on difficult-to-classify samples |
| Regularization | Learning rate and tree depth provide effective regularization |

#### 3.2.3 Health Score Computation

Beyond binary classification, a continuous health score (1-10 scale) was implemented:

$$H_{score} = 10 - (P_{unhealthy} \times 5) - (R_{severity} \times 5)$$

Where:
- $P_{unhealthy}$: Model's predicted probability of unhealthy status
- $R_{severity}$: Mean of active risk indicators (0-1 scale)

This scoring mechanism provides:
- **Granular Assessment**: Distinguishes between marginally and severely degraded equipment
- **Actionable Insights**: Enables prioritization of maintenance activities
- **Trend Monitoring**: Facilitates tracking of equipment health over time

### 3.3 Regression: Remaining Useful Life Estimation

#### 3.3.1 Problem Formulation

RUL estimation is formulated as a regression problem:

$$RUL = f(X) \in \mathbb{R}^+$$

Where RUL represents the predicted operational hours remaining before maintenance is required.

#### 3.3.2 Degradation-Adjusted RUL

A key contribution of this work is the development of a degradation-adjusted RUL model that accounts for operating conditions:

$$RUL_{adjusted} = \frac{RUL_{base}}{D_{rate}}$$

Where:
- $RUL_{base} = W_{max} - W_{current}$ (maximum tool wear minus current tool wear)
- $D_{rate}$ = Degradation rate multiplier (≥ 1.0)

**Degradation Rate Computation**

The degradation rate captures accelerated wear under stress conditions:

$$D_{rate} = 1.0 + \sum_{i} \alpha_i \cdot S_i$$

Where $S_i$ represents individual stress factors:

| Stress Factor | Condition | Weight ($\alpha$) |
|---------------|-----------|-------------------|
| Thermal Stress | Deviation from optimal ΔT | 0.30 |
| Power Stress | Power outside 4,000-8,000W | 0.40 |
| Torque Stress | Torque > 50 Nm | 0.30 |
| RPM Stress | RPM < 1,300 or > 2,000 | 0.20 |
| Risk Factor | Per active risk indicator | 0.25 |

#### 3.3.3 Model Selection: Gradient Boosting Regressor

**Justification for Selection**

| Criterion | Gradient Boosting Advantage |
|-----------|----------------------------|
| Non-linear Relationships | Captures complex degradation patterns |
| Feature Interactions | Models interactions between stress factors |
| Robustness | Handles outliers in RUL values |
| Accuracy | State-of-the-art performance on tabular regression |

---

## 4. Model Evaluation

### 4.1 Evaluation Metrics

#### 4.1.1 Classification Metrics

**Accuracy**
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision**
$$Precision = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)**
$$Recall = \frac{TP}{TP + FN}$$

**F1-Score**
$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

#### 4.1.2 Regression Metrics

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Root Mean Square Error (RMSE)**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of Determination (R²)**
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

### 4.2 Results Summary

#### 4.2.1 Failure Type Prediction (Multi-Class)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 95.70% |
| Cross-Validation Accuracy | 98.28% ± 0.28% |
| Macro F1-Score | 0.68 |
| Weighted F1-Score | 0.97 |

**Per-Class Performance**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Failure | 1.00 | 0.96 | 0.98 | 1,930 |
| Tool Wear Failure | 0.09 | 0.56 | 0.16 | 9 |
| Heat Dissipation Failure | 1.00 | 1.00 | 1.00 | 23 |
| Power Failure | 1.00 | 1.00 | 1.00 | 18 |
| Overstrain Failure | 0.89 | 1.00 | 0.94 | 16 |
| Random Failure | 0.00 | 0.00 | 0.00 | 4 |

#### 4.2.2 Health Classification (Binary)

| Metric | Value |
|--------|-------|
| Accuracy | 99.90% |
| Cross-Validation Accuracy | 99.90% ± 0.06% |
| Precision (Unhealthy) | 1.00 |
| Recall (Unhealthy) | 0.99 |
| F1-Score (Unhealthy) | 1.00 |

#### 4.2.3 RUL Estimation (Regression)

| Metric | Value |
|--------|-------|
| Mean Absolute Error | 0.28 hours |
| Root Mean Square Error | 0.75 hours |
| R² Score | 0.9999 |
| Cross-Validation R² | 0.9999 ± 0.0000 |

**RUL Category Accuracy**

| Category | Range | Accuracy |
|----------|-------|----------|
| Critical | < 20 hours | 92.3% |
| Warning | 20-50 hours | 99.4% |
| Moderate | 50-100 hours | 99.8% |
| Good | > 100 hours | 99.9% |

### 4.3 Feature Importance Analysis

The feature importance analysis validates the effectiveness of physics-based feature engineering:

**Failure Type Prediction**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | PWF Risk | 0.1358 |
| 2 | Risk Score | 0.1223 |
| 3 | HDF Risk | 0.1212 |
| 4 | OSF Risk | 0.0923 |
| 5 | Strain | 0.0866 |

**Health Classification**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Risk Count | 0.9928 |
| 2 | Power | 0.0018 |
| 3 | Strain | 0.0012 |

The dominance of engineered risk indicators in feature importance rankings confirms that physics-based features effectively capture the underlying failure mechanisms.

---

## 5. Discussion

### 5.1 Model Selection Trade-offs

The selection of ensemble methods (Random Forest and Gradient Boosting) over deep learning approaches was driven by several practical considerations:

1. **Dataset Size**: With 10,000 samples, the dataset is insufficient to train deep neural networks without overfitting
2. **Interpretability**: Ensemble methods provide feature importance rankings essential for maintenance decision-making
3. **Training Efficiency**: Ensemble methods train in seconds compared to hours for deep learning
4. **Deployment Simplicity**: Scikit-learn models are easily serialized and deployed without GPU requirements

### 5.2 Handling Extreme Class Imbalance

The Random Failure class (0.19% of data) remains challenging despite SMOTE application. With only 18 total samples, the class lacks sufficient representation for reliable pattern learning. Potential solutions include:

1. Collecting additional failure data
2. Treating Random Failure as an anomaly detection problem
3. Combining with No Failure class for practical deployment

### 5.3 Practical Implications

The three-model architecture provides complementary information for maintenance decision-making:

| Model | Decision Support |
|-------|------------------|
| Failure Type | Specific failure mode enables targeted maintenance actions |
| Health Score | Prioritization of equipment for inspection |
| RUL | Scheduling of maintenance windows |

---

## 6. Conclusion

This report presented a comprehensive predictive maintenance system employing three machine learning models optimized for distinct predictive tasks. The key contributions include:

1. **Physics-based feature engineering** that captures domain knowledge of failure mechanisms
2. **SMOTE-based class balancing** for handling severe class imbalance in failure data
3. **Degradation-adjusted RUL** that accounts for operating stress conditions
4. **Multi-model architecture** providing complementary maintenance insights

The system achieves high accuracy across all tasks: 95.70% for multi-class failure prediction, 99.90% for health classification, and R²=0.9999 for RUL estimation. These results demonstrate the effectiveness of combining ensemble learning methods with physics-informed feature engineering for industrial predictive maintenance applications.

---

## References

1. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
2. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189-1232.
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
4. AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository.
5. Carvalho, T. P., et al. (2019). A systematic literature review of machine learning methods applied to predictive maintenance. Computers & Industrial Engineering, 137, 106024.

---

## Appendix A: Model Hyperparameters

### A.1 Random Forest (Failure Prediction)

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
```

### A.2 Gradient Boosting (Health Classification)

```python
GradientBoostingClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### A.3 Gradient Boosting (RUL Estimation)

```python
GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

---

## Appendix B: System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Sensor Data                       │
│  (Temperature, RPM, Torque, Tool Wear, Product Type)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Feature Engineering Layer                      │
│  • Physics-based features (ΔT, Power, Strain)               │
│  • Binary risk indicators (HDF, PWF, OSF, TWF)              │
│  • Degradation rate computation                              │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────┐
│ Failure Type      │ │ Health        │ │ RUL           │
│ Prediction        │ │ Classification│ │ Estimation    │
│ (Random Forest)   │ │ (Grad Boost)  │ │ (Grad Boost)  │
└───────────────────┘ └───────────────┘ └───────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
┌───────────────────┐ ┌───────────────┐ ┌───────────────┐
│ • Failure Type    │ │ • Health/     │ │ • Hours       │
│ • Confidence %    │ │   Unhealthy   │ │   Remaining   │
│ • Top 3 Probs     │ │ • Score (1-10)│ │ • Degradation │
│                   │ │ • Risk Details│ │   Rate        │
└───────────────────┘ └───────────────┘ └───────────────┘
```

---

*Report generated for academic purposes. All models and methodologies are implemented in the accompanying codebase.*
