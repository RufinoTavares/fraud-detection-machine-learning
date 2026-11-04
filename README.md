# Fraud Detection Using Machine Learning

**End-to-End Data Science Project | Python • Machine Learning • Fraud Analytics**

## 1. Project Overview

Financial transaction fraud is a challenging classification problem because fraudulent transactions represent only a small proportion of the total volume. An effective detection system must identify suspicious activity while controlling the number of legitimate transactions incorrectly flagged as fraud.

This project develops a machine learning approach to identify potentially fraudulent transactions using historical data from 2025 and generate predictions for January 2026.

The analysis covers the complete data science workflow, from exploratory data analysis and feature engineering to model evaluation, threshold selection, and business recommendations.

**Main objectives:**

- Investigate transaction characteristics associated with fraudulent behavior.
- Identify relevant patterns and potential risk indicators.
- Develop and compare supervised classification models.
- Improve model performance through temporal feature engineering.
- Evaluate the trade-off between fraud detection and false-positive alerts.
- Apply the selected model to new, unlabeled transactions.

## 2. Dataset Description

The historical dataset contains **523,525 transactions recorded during 2025**, including legitimate and fraudulent transactions.

| Dataset characteristic | Value |
|---|---|
| Historical transactions | 523,525 |
| Fraudulent transactions | 20,485 |
| Legitimate transactions | 503,040 |
| Historical fraud rate | 3.91% |
| Missing values | None |
| Duplicate records | None |
| New transactions for prediction | 44,451 |

The target variable, `flag_fraude`, indicates whether a transaction was classified as fraudulent.

One of the main challenges is the **class imbalance**: approximately 96% of transactions are legitimate, meaning that accuracy alone is not sufficient to evaluate fraud detection performance.

## 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the distribution of transactions, investigate fraud patterns, and identify variables that could contribute to predictive performance.

### 3.1 Fraud Distribution

Fraudulent transactions account for only 3.91% of the historical dataset.

This imbalance motivates the use of evaluation metrics such as precision, recall, F1-score, and Average Precision rather than relying exclusively on accuracy.

### 3.2 New Devices and Fraud Risk

Transactions performed on new devices exhibit substantially higher fraud rates.

| Device status | Fraud rate |
|---|---|
| New device | 12.90% |
| Previously used device | 2.13% |

Fraud is approximately six times more frequent among transactions made on new devices.

This finding suggests that device novelty is a relevant risk indicator, although it should not be used as an isolated rule for classifying transactions.

### 3.3 Fraud by Establishment Type

Fraud rates also vary across establishment categories.

| Establishment type | Fraud rate |
|---|---|
| Fuel | 7.55% |
| Food & Beverage | 6.63% |
| E-commerce | 3.12% |
| Retail | 2.51% |
| Supermarket | 2.37% |
| Pharmacy | 2.33% |

Fuel and Food & Beverage establishments have the highest observed fraud rates.

These differences suggest that transaction context can contribute to fraud risk assessment.

### 3.4 Temporal Fraud Patterns

Transaction timing emerged as one of the most informative characteristics in the analysis.

Fraud rates are particularly high during late-night hours:

- Approximately 30% at 10 PM and 11 PM.
- Approximately 6.7%–6.9% between 12 PM and 2 PM.
- Considerably lower fraud rates during several morning and afternoon periods.

These findings indicate a strong association between transaction hour and fraudulent activity in the historical dataset.

The temporal patterns motivated the inclusion of additional time-related features in the predictive model.

### 3.5 Transaction Amount and Internal Score

Transaction amount and internal score were also investigated.

The average transaction amount was approximately 44.50 for legitimate transactions and 30.86 for fraudulent transactions.

The internal score showed very similar distributions across both classes, with average values of approximately 500.42 and 499.34, respectively.

These findings indicate that individual variables may provide limited discrimination, reinforcing the importance of combining multiple transaction characteristics.

## 4. Data Preparation and Feature Engineering

The dataset was prepared for supervised classification, including feature selection, categorical variable processing, and temporal validation.

### 4.1 Chronological Data Splitting

A chronological approach was used to separate training, validation, and test data.

| Dataset | Period |
|---|---|
| Training | January–August 2025 |
| Validation | September–October 2025 |
| Test | November–December 2025 |
| Prediction | January 2026 |

This strategy simulates a realistic scenario in which a model is trained using historical information and evaluated on later transactions.

The validation set was used for model development and threshold selection, while the test set was reserved for final evaluation.

### 4.2 Temporal Feature Engineering

Additional features were derived from transaction timestamps, including:

- Transaction hour.
- Day of the week.

These features were introduced to capture time-dependent fraud patterns that were not fully represented by the original variables.

The final model used 12 predictive features, compared with 10 in the earlier configuration.

## 5. Machine Learning Models

Two supervised classification algorithms were evaluated.

### 5.1 Logistic Regression

Logistic Regression was used as a baseline model because it provides an interpretable reference for binary classification.

Its performance was evaluated using classification metrics appropriate for imbalanced datasets.

### 5.2 Random Forest

Random Forest was evaluated as a more flexible alternative capable of capturing nonlinear relationships and interactions among transaction characteristics.

The Random Forest approach was selected for further development.

### 5.3 Model Improvement

A comparison was conducted between Random Forest configurations using 10 and 12 predictive features.

| Model configuration | Validation Average Precision |
|---|---|
| Random Forest — 10 features | 0.4309 |
| Random Forest — 12 features | 0.6362 |

The inclusion of temporal features increased validation Average Precision by approximately **47.7%**.

This improvement demonstrates the predictive value of transaction timing in the analyzed dataset.

## 6. Model Evaluation

Because fraudulent transactions are relatively rare, the evaluation focused on metrics that capture both the ability to detect fraud and the reliability of positive predictions.

The main evaluation metrics were:

- **Precision:** proportion of flagged transactions that are actually fraudulent.
- **Recall:** proportion of actual fraudulent transactions successfully detected.
- **F1-score:** harmonic mean of precision and recall.
- **Average Precision (AP):** summary of the precision–recall relationship across classification thresholds.
- **Confusion matrix:** distribution of correct predictions, false positives, and false negatives.

### Final Model Performance

The final Random Forest model achieved the following results on the November–December 2025 test set:

| Metric | Result |
|---|---|
| Average Precision | 0.6164 |
| Precision | 50.64% |
| Recall | 74.42% |
| F1-score | 0.6027 |
| Classification threshold | 0.80 |

At the selected threshold, the model detected approximately 74% of fraudulent transactions in the test set.

The precision of approximately 51% indicates that a substantial proportion of generated alerts would still require investigation.

## 7. Classification Threshold and Business Trade-offs

The classification threshold determines the probability level at which a transaction is flagged as potentially fraudulent.

A lower threshold generally increases fraud detection sensitivity but may also produce more false-positive alerts.

A higher threshold can reduce the investigation workload but may allow more fraudulent transactions to go undetected.

The threshold of **0.80** was selected using validation performance, with F1-score as the main selection criterion.

This threshold is recommended as an initial operating point because it provides a practical balance between precision and recall in the evaluated model.

However, the optimal operational threshold would depend on additional business information, including:

- Financial losses associated with undetected fraud.
- Costs of investigating false-positive alerts.
- Investigation team capacity.
- Customer experience and transaction friction.

A threshold of 0.50 can be considered as an alternative when the business objective prioritizes higher recall.

## 8. January 2026 Fraud Predictions

After model selection and evaluation, the final pipeline was applied to the unlabeled January 2026 dataset.

| Prediction summary | Result |
|---|---|
| Total transactions | 44,451 |
| Transactions flagged as suspicious | 3,168 |
| Predicted alert rate | 7.13% |
| Selected threshold | 0.80 |

The model identified 3,168 transactions requiring potential investigation.

**Important:** these alerts are model predictions, not confirmed fraudulent transactions.

The predicted alert rate should therefore not be interpreted as the actual fraud rate for January 2026.

## 9. Business Insights and Recommendations

The analysis provides several findings relevant to a potential fraud prevention strategy.

**Transaction timing is a strong predictive signal.** Late-night transactions showed substantially higher historical fraud rates, and adding temporal features improved model performance.

**Device novelty is associated with higher risk.** Transactions made on new devices exhibited a fraud rate approximately six times higher than transactions made on previously used devices.

**Fraud risk depends on transaction context.** Establishment type and other transaction characteristics contribute information that can support classification.

**Model predictions can help prioritize investigations.** Rather than automatically treating every alert as confirmed fraud, the system can support risk-based transaction review.

### Recommended Operational Approach

1. Use the Random Forest model with a threshold of 0.80 as an initial configuration.
2. Prioritize suspicious transactions for additional verification.
3. Monitor precision, recall, alert volume, and changes in transaction patterns.
4. Reevaluate the threshold using actual fraud losses and investigation costs.
5. Periodically retrain and validate the model using more recent transactions.

## 10. Limitations and Future Improvements

Although the model achieved promising results, several limitations should be considered.

- Model performance was evaluated using historical data and may change over time.
- Fraud patterns may evolve, requiring continuous monitoring.
- False-positive alerts can create additional investigation workload.
- January 2026 predictions do not have confirmed labels in the provided dataset.
- The optimal business threshold cannot be determined without real operational cost information.
- Additional customer behavioral features may improve detection performance.

Potential improvements include customer-level transaction history, more detailed behavioral features, probability calibration, model monitoring, and evaluation of investigation costs.

## 11. Technologies and Tools

- **Programming:** Python
- **Data manipulation:** pandas, NumPy
- **Data visualization:** Matplotlib
- **Machine Learning:** scikit-learn
- **Models:** Logistic Regression, Random Forest
- **Environment:** Google Colab, Jupyter Notebook
- **Evaluation:** Precision, Recall, F1-score, Average Precision, Confusion Matrix

## 12. Project Files

**Jupyter Notebook:** [Fraud Detection Analysis](ATF_Case_Fraude_MicheleTavares_Final.ipynb)

The notebook contains the exploratory analysis, feature engineering, model development, evaluation, and prediction workflow.

The original assessment datasets are not included in this repository.

## Author

**Michele Tavares**

Background in Computational Physics, with an interest in Data Science, Machine Learning, statistical analysis, and practical applications of predictive modeling.

---

*This repository is maintained privately. The project was developed as part of a technical assessment, and the original assessment materials are not publicly distributed.*
