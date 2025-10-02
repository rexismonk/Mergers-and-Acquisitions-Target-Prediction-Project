# M&A Target Prediction Project: A Quantitative Approach to Deal Sourcing

## Project Overview

This project develops a quantitative framework to predict M&A (Mergers and Acquisitions) targets among venture-funded companies. The core of this initiative is a gradient boosting classification model trained on a real-world dataset of over 30,000 firms. By identifying the key financial and operational drivers of acquisitions, this project serves as a proof-of-concept for a tool that can provide a significant data-driven edge in the traditionally relationship-driven field of deal sourcing.

For an investment bank or corporate development team, this model can automate and enhance the target screening process, shifting it from a manual effort to a highly efficient, analytical workflow that uncovers both obvious and non-obvious opportunities.

## Data & Engineering

The model's foundation was built by fusing two disparate, real-world datasets:

* **Company Features:** A comprehensive list of companies, their funding history, market sector, and operational timeline from `investments_VC.csv`.
* **Acquisition Data:** A transactional list of historical M&A deals from `Acquisitions.csv`.

The primary data engineering challenge involved robust string normalization using regex to accurately merge company names across these distinct sources.

## Technical Workflow & Methodology

The project followed a rigorous, multi-stage machine learning pipeline:

1.  **Feature Engineering:** Predictive features were engineered to capture key business concepts. This included creating temporal features like `company_age_days` to model firm maturity—a critical factor in M&A valuation and target attractiveness.

2.  **Predictive Modeling & Imbalance Handling:**
    * **Task:** The problem was framed as a binary classification task to predict a `1` (acquired within one year) or `0` (not acquired).
    * **Imbalance:** The target class was extremely rare, creating a severe class imbalance. This was addressed using **SMOTE (Synthetic Minority Over-sampling TEchnique)** on the training data to prevent the model from simply ignoring the positive class and achieving a high but useless accuracy score.
    * **Models:** A Logistic Regression with L2 regularization served as a robust baseline. The primary model was an optimized **XGBoost (eXtreme Gradient Boosting) classifier**, renowned for its high performance and scalability on structured, tabular data.

3.  **Model Validation & Explainable AI (XAI):**
    * **Evaluation:** Model performance was rigorously validated on a held-out test set using **ROC AUC**, a metric resilient to class imbalance, alongside precision and recall to assess real-world utility in a business context.
    * **Interpretation:** To move beyond "black box" predictions, **SHAP (SHapley Additive exPlanations)** was employed. This game-theory-based approach decomposes the model's output, attributing prediction drivers to each feature for full transparency and the generation of proprietary business insights.

## Results & Key Findings

The final XGBoost model demonstrated significant predictive power, achieving a **ROC AUC score of 0.66 on the test set. This indicates a strong ability to differentiate between future M&A targets and non-targets.

The SHAP analysis unlocked the following actionable insights, translating the model's logic into investment banking strategy:

1.  **Firm Maturity (`company_age_days`):** This was the single most dominant predictor. Mature, more established companies are prime acquisition targets, likely due to their stable operations and proven market fit.
2.  **Capitalization & Investor Validation (`funding_total_usd`):** Higher total funding acts as a strong proxy for quality and de-risking. Companies that have successfully raised significant capital are far more likely to be acquired.
3.  **Track Record (`funding_rounds`):** The number of funding rounds serves as a measure of a company's ability to consistently hit milestones, making it a more attractive and validated target.
4.  **Market Focus (`market_simplified`):** Companies in mainstream, well-understood sectors (e.g., Software, Biotechnology) were more likely targets than those in niche markets, suggesting that a larger pool of potential strategic acquirers is a key factor.

## How to Run This Project

1.  Clone the repository.
2.  Install the required dependencies (`pandas`, `scikit-learn`, `xgboost`, `imblearn`, `shap`).
3.  Place the raw CSV files from the original data source into the `data/raw/` directory.
4.  Run the Jupyter Notebooks in sequential order from `01` to `04` to replicate the analysis.
