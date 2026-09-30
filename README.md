# Predictive Modeling of Career Seniority in the Global AI & Data Job Market

A machine learning pipeline that predicts the **experience level** (Entry, Mid, Senior, Lead) of an AI/Data job listing from features such as salary, job role, company size, and work conditions.

Course project, Dept. of Computer Science, BRAC University.
**Authors:** Aryan Reza, Shahriar Mohammad

## Problem
Multi-class classification: given a job listing's attributes, predict its `experience_level`. This can help recruiters gauge the right seniority for a role and help job seekers benchmark compensation.

## Dataset
- Global AI & Data Jobs Salary Dataset: **90,000 records**
- Perfectly balanced target (~22,500 per class), no missing values
- Features include salary, bonus, job role, country, company size, education, work mode, weekly hours, work-life balance, and more

## Pipeline
1. **EDA:** correlation matrix, median salary by job role, class balance checks
2. **Preprocessing:**
   - Dropped `id` and duplicate rows
   - Label-encoded the target (0-3)
   - One-hot encoded categorical features (`pd.get_dummies`)
   - Removed leakage-prone features such as `salary_percentile`
   - Stratified 80/20 train-test split
   - `StandardScaler` for numerical features
3. **Modeling:** five classifiers trained and compared
4. **Evaluation:** accuracy plus macro-averaged precision, recall, and F1; confusion matrices and one-vs-rest ROC curves

## Models & Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| Logistic Regression | 0.9987   | 0.9987    | 0.9987 | 0.9987   |
| Random Forest       | 0.9971   | 0.9971    | 0.9970 | 0.9970   |
| KNN (k = 15)        | 0.7812   | 0.7862    | 0.7817 | 0.7825   |
| AdaBoost            | 0.4974   | 0.3329    | 0.5000 | 0.3746   |

**Hyperparameters:** Decision Tree `max_depth=10`; Random Forest `n_estimators=200, max_depth=10`; AdaBoost `n_estimators=200, learning_rate=0.5`; KNN `k=15` chosen via 5-fold cross-validation.

## Key Findings
- Random Forest feature importance shows `experience_years` and `salary_usd` (along with `layoff_risk` and `job_security_score`) as the strongest predictors of seniority.
- Distance-based KNN degrades with the high dimensionality introduced by one-hot encoding.
- Near-perfect scores from the tree models and Logistic Regression suggest the dataset is **synthetic** with very clean separability, so results may not transfer to real job-board data.

## Limitations & Future Work
- Validate on imbalanced, scraped data from live job boards
- Re-run with `experience_years` removed to test how much it drives the result
- Tune AdaBoost (e.g., deeper base learners) and try gradient boosting

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib / Seaborn

## Repository Structure
```
├── data/            # dataset (or download link)
├── notebooks/       # EDA and model training
├── figures/         # confusion matrices, ROC curves, feature importance
├── report.pdf       # full project report
└── README.md
```