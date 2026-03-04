"""
Customer Cluster Predictor
--------------------------
Author: Rohan Neupane
Email:  rohannneupane02@gmail.com

Predicts which customer segment a new customer belongs to,
based on Annual Income and Spending Score.

Usage:
    python predict_cluster.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import os

# Path to dataset (relative to this file)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Mall_Customers.csv')

# What each cluster number means
CLUSTER_DESCRIPTIONS = {
    0: "Cluster 1 (Green)  — High Income, Low Spending   → Careful spender",
    1: "Cluster 2 (Red)    — High Income, High Spending  → Prime target customer ⭐",
    2: "Cluster 3 (Yellow) — Average Income & Spending   → Standard customer",
    3: "Cluster 4 (Violet) — Low Income, High Spending   → Impulsive spender",
    4: "Cluster 5 (Blue)   — Low Income, Low Spending    → Budget-conscious saver",
}


def train_model():
    """Train KMeans on the mall customer dataset and return the model."""
    customer_data = pd.read_csv(DATA_PATH)
    X = customer_data.iloc[:, [3, 4]].values

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
    kmeans.fit(X)
    return kmeans


def predict_cluster(annual_income, spending_score, model):
    """
    Predict which customer segment a new customer belongs to.

    Parameters:
        annual_income  (float): Annual income in k$
        spending_score (int):   Spending score between 1-100
        model:                  Trained KMeans model

    Returns:
        dict: cluster number and description
    """
    input_data = np.array([[annual_income, spending_score]])
    cluster = model.predict(input_data)[0]
    return {
        'cluster_number': cluster + 1,
        'description': CLUSTER_DESCRIPTIONS[cluster]
    }


if __name__ == '__main__':
    print("=" * 58)
    print("   Mall Customer Segmentation — Rohan Neupane")
    print("=" * 58)

    # Train the model on full dataset
    model = train_model()
    print("✅ Model trained on 200 customers.\n")

    # Try some example customers
    examples = [
        {"annual_income": 15,  "spending_score": 39,  "label": "Low income, low spender"},
        {"annual_income": 99,  "spending_score": 97,  "label": "High income, high spender"},
        {"annual_income": 55,  "spending_score": 49,  "label": "Average income & spending"},
        {"annual_income": 17,  "spending_score": 81,  "label": "Low income, high spender"},
        {"annual_income": 137, "spending_score": 18,  "label": "High income, low spender"},
    ]

    for customer in examples:
        result = predict_cluster(
            customer["annual_income"],
            customer["spending_score"],
            model
        )
        print(f"Customer : {customer['label']}")
        print(f"  Income : {customer['annual_income']}k$  |  Spending Score : {customer['spending_score']}")
        print(f"  Segment: {result['description']}")
        print()
