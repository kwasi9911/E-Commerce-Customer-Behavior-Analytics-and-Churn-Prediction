# E-Commerce Customer Behavior Analytics & Churn Prediction

A cloud-native data science pipeline on **Google Cloud Platform** that cleans, analyzes, and models e-commerce transaction data to predict customer churn — built end-to-end in BigQuery and BigQuery ML.

## Overview

Using the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (~1M transactions, UK-based online retailer, 2009–2011), this project:

- Ingests and cleans transactional data entirely in BigQuery SQL
- Explores purchasing patterns, seasonality, and customer geography
- Engineers an RFM (Recency, Frequency, Monetary) feature table per customer
- Trains and compares three classifiers in-database with BigQuery ML
- Evaluates models on a held-out year and analyzes the churn-probability decision threshold

## Tech Stack

`Google Cloud Platform` · `BigQuery` · `BigQuery ML` · `Google Cloud Storage` · `Python (Pandas, Matplotlib, Seaborn, scikit-learn)`

## Pipeline

| Stage | What happens |
|---|---|
| **Ingestion** | Raw Excel → CSV, staged in GCS, loaded into BigQuery with explicit schemas |
| **Cleaning** | SQL removes null CustomerIDs, cancellations, and invalid price/quantity rows → `retail_clean` table |
| **EDA** | Order value distribution, top products, day/hour seasonality, monthly revenue trend, country breakdown, inter-purchase intervals |
| **Feature Engineering** | Per-customer RFM + return-rate + product-diversity features, materialized for a 2010 training window and a 2011 hold-out window |
| **Modeling** | `CREATE MODEL` in BigQuery ML: Logistic Regression, Gradient Boosted Trees, DNN Classifier |
| **Evaluation** | `ML.EVALUATE`, `ML.ROC_CURVE`, `ML.FEATURE_IMPORTANCE`, `ML.WEIGHTS` on the hold-out set |
| **Threshold Analysis** | Precision/recall/F1 sweep across probability thresholds to find the optimal churn-flag cutoff |

## Key Results

After cleaning, the dataset had **805,549 transactions** across **5,878 customers**. The training feature table showed a **37% churn rate** under a 90-day no-purchase definition.

All three models scored near-perfect on the hold-out set (AUC-ROC ≈ 1.0). However, the GBT feature importance output assigns essentially all predictive weight to `recency_days` alone — which is expected, since the churn label is *defined* as "no purchase within 90 days," making recency a direct restatement of the label rather than an independent predictor. **This is label leakage, not genuine model skill**, and is the most important caveat in this project. A more defensible version of this pipeline would predict churn using only features observable *before* the recency threshold is crossed (e.g., trend in purchase frequency, early return behavior), holding out recency itself as a feature.

## Repository Contents

- `E-Commerce_Customer_Behavior_Analytics_and_Churn_Prediction.ipynb` — full notebook (setup → ingestion → cleaning → EDA → feature engineering → modeling → evaluation → threshold analysis)

## Setup

1. Set `PROJECT_ID`, `DATASET_ID`, and `GCS_BUCKET` in the config cell
2. Upload `online_retail_ii.csv` (converted from the original `.xlsx`) to your GCS bucket
3. Run all cells top to bottom — every transformation is a `CREATE OR REPLACE TABLE`/`MODEL` statement, so the pipeline is fully reproducible from raw data

## Future Work

- Remove `recency_days` from the feature set (or redefine the churn label independently of it) to get an honest measure of predictive power
- Deploy the trained model behind a Vertex AI Prediction endpoint and a Cloud Run web app for interactive threshold-based predictions
- Incorporate the Olist dataset more deeply for geographic and category-level features
