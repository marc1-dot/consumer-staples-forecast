# Consumer Staples Forecasting

## 🧠 Project Overview
This project develops a **machine learning pipeline to forecast weekly stock returns** and evaluate **trading strategies** in the **Consumer Staples sector**.

The analysis focuses on five major publicly traded companies:
- **Nestlé (NESN.SW)**
- **Procter & Gamble (PG)**
- **Unilever (UL)**
- **Coca-Cola (KO)**
- **PepsiCo (PEP)**

Several machine learning models are trained and compared, and their performance is evaluated both **statistically** (out-of-sample prediction accuracy) and **economically** (backtesting and Monte Carlo simulations).

The project follows academic best practices for **time-series modeling**, **data leakage prevention**, and **out-of-sample evaluation**.

---

## 🎯 Objectives
1. Automatically collect historical market data.
2. Clean and preprocess financial time-series data.
3. Engineer predictive features suitable for return forecasting.
4. Train multiple machine learning models (Linear Regression, Neural Network, Random Forest, XGBoost).
5. Evaluate models using a strict temporal train/validation/test split (**70% Train / 20% Validation / 10% Test**).
6. Backtest trading strategies based on model predictions.
7. Assess robustness using Monte Carlo simulations.

---

## 🧰 Technologies & Requirements

### System Requirements
- **Python**: 3.12.11
- **Platform**: Nuvolos or Linux-based environment
- **RAM**: Minimum 8GB recommended
- **Disk Space**: ~500MB for data and results

### Core Dependencies
```text
pandas==2.3.2
numpy==2.3.2
scikit-learn==1.7.1
xgboost==3.1.2
matplotlib==3.10.5
scipy==1.16.1
seaborn==0.13.2
yfinance

##📁 Repository Structure

    
consumer-staples-forecasting/
│
├── main.py                                  # MAIN ENTRY POINT (Runs full pipeline)
├── data/                                    # Data directory
│   ├── raw/                                 # Raw downloaded data
│   │   └── consumer_staples_data.csv
│   └── processed/                           # Processed splits
│       ├── train.csv                        # Training set (70%)
│       ├── validation.csv                   # Validation set (20%)
│       └── test.csv                         # Test set (10%)
│
├── src/                                     # Source code
│   ├── __init__.py
│   ├── data_loader.py                       # Step 1: Data acquisition
│   ├── preprocessing.py                     # Step 2: Cleaning & Engineering
│   ├── create_train_validation_test_split.py# Step 3: Temporal split + look ahead biais
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linear_model.py
│   │   ├── neural_network.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   ├── model_evaluate.py                    # Step 4: Initial evaluation
│   ├── train_all.py                         # Step 5: Model training
│   ├── test_all.py                          # Step 6: Final testing
│   ├── feature_importance.py                # Step 7: Feature analysis
│   ├── backtesting.py                       # Step 8: Backtesting strategies
│   ├── monte_carlo.py                       # Step 9: Risk simulation
│   ├── linear_regression_model.py           # Model class
│   ├── neural_network_model.py              # Model class
│   ├── random_forest_model.py               # Model class
│   └── xgboost_model.py                     # Model class
│
├── results/                                 # Generated outputs
│   ├── models/                              # Trained models (.pkl)
│   ├── figures/                             # Visualizations (.png)
│   └── metrics/                             # Performance metrics (.csv)
│
├── environment.yml                          # Conda environment specification
├── requirements.txt                         # pip dependencies
└── README.md                                # This file


##🚀 Setup & Installation

## Installation

To set up the environment, run the following commands:

conda env create 
-f environment.yml -n consumer-staples-forecast

##🎬 Execution Instructions

python main.py


##📊 Expected OutputsAfter running the pipeline, check the results/ folder:
Models: 4 .pkl files in results/models/
Metrics: test_performance.csv (Neural Network is expected to have the highest R² ~0.138)
Figures: Backtesting charts, Monte Carlo histograms, and feature importance plots in results/figures/

📄 LicenseAcademic project for Advanced Programming - HEC Lausanne (Fall 2025).
Data sourced from Yahoo Finance.