# E-Commerce Customer Intent Segmentation

This repository contains a machine learning project that segments e-commerce customers based on their browsing behavior and intent. Using K-Means clustering, the project categorizes users into high-intent purchasers and high-risk window shoppers by analyzing two key metrics: **Time on Site** and **Cart Abandonment Rate**.

## Project Overview

Understanding customer intent is crucial for optimizing e-commerce conversion rates. This project simulates an environment where user engagement metrics are used to predict purchase likelihood. It includes a synthetic data generator to create a reliable dataset and an analysis script that applies unsupervised learning (K-Means) to identify distinct customer cohorts. The clusters are then evaluated against actual purchase outcomes using the Adjusted Rand Index and a Confusion Matrix.

## Features

* **Custom Data Generation:** A lightweight, built-in script to generate a 500-row synthetic dataset (`custom_shopping_data.csv`) featuring realistic distributions of customer behavior.
* **Unsupervised Clustering:** Implementation of K-Means clustering with feature scaling (`StandardScaler`) to balance percentage-based and time-based metrics.
* **Optimal Cluster Validation:** Automated generation of an Elbow Method plot to visually verify the optimal number of clusters.
* **Data Visualization:** Clean, modern visualizations using `matplotlib` and `seaborn` to map customer segments and display the confusion matrix.
* **Performance Evaluation:** Uses Adjusted Rand Index (ARI) to measure how well the unsupervised clusters align with actual purchase data.

## Prerequisites

To run this project, you will need Python 3.7+ and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

```

## Repository Structure

* `create_data.py`: The data generation script. Run this first to build the dataset.
* `analyze_data.py`: The core machine learning script that loads the data, scales features, runs K-Means, and visualizes the results.
* `custom_shopping_data.csv`: The generated dataset (appears after running `create_data.py`).

## Usage

**1. Generate the Dataset**
Run the data creation script to generate the synthetic customer records. This will output a file named `custom_shopping_data.csv` in your active directory.

```bash
python create_data.py

```

**2. Run the Analysis**
Execute the main analysis script. This will process the CSV, perform the clustering, and output several visualizations to your screen.

```bash
python analyze_data.py

```

## What to Expect (Visual Outputs)

When you run the analysis script, it will generate three sequential plots:

1. **The Elbow Method Plot:** Confirms that 2 clusters is the mathematically optimal choice for this dataset.
2. **Customer Segmentation Scatter Plot:** A 2D mapping of all 500 users, color-coded by their assigned cluster (High Intent vs. High Risk).
3. **Confusion Matrix Heatmap:** Evaluates the accuracy of the unsupervised clustering by comparing the predicted clusters against the actual `Made_Purchase` column.
