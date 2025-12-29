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
5. Evaluate models using a strict temporal train/validation/test split (70%/20%/10%).
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
pandas==2.3.2
numpy==2.3.2
scikit-learn==1.7.1
xgboost==3.1.2
matplotlib==3.10.5
scipy==1.16.1
seaborn==0.13.2
yfinance

---

## 📁 Repository Structure

consumer-staples-forecasting/
│
├── data/                                    # Data directory
│   ├── raw/                                 # Raw downloaded data
│   │   └── consumer_staples_data.csv        # Original dataset
│   └── processed/                           # Processed splits
│       ├── train.csv                        # Training set (70%)
│       ├── validation.csv                   # Validation set (20%)
│       └── test.csv                         # Test set (10%)
│
├── src/                                     # Source code
│   ├── download_data.py                     # Step 1: Data acquisition
│   ├── create_train_validation_test_split.py # Step 2: Temporal split
│   ├── train_all.py                         # Step 3: Model training
│   ├── test_all.py                          # Step 4: Model evaluation
│   ├── backtesting.py                       # Step 5: Backtesting
│   ├── monte_carlo.py                       # Step 6: Monte Carlo simulations
│   ├── linear_regression_model.py           # Linear Regression implementation
│   ├── neural_network_model.py              # Neural Network implementation
│   ├── random_forest_model.py               # Random Forest implementation
│   └── xgboost_model.py                     # XGBoost implementation
│
├── results/                                 # Generated outputs
│   ├── models/                              # Trained models (.pkl)
│   │   ├── linear_regression_model.pkl
│   │   ├── neural_network_model.pkl
│   │   ├── random_forest_model.pkl
│   │   └── xgboost_model.pkl
│   ├── figures/                             # Visualizations (.png)
│   │   ├── backtesting_comparison_*.png     # Backtesting results
│   │   ├── monte_carlo_*.png                # Monte Carlo histograms
│   │   ├── model_residuals.png              # Residual analysis
│   │   └── r2_comparison.png                # Model performance
│   └── metrics/                             # Performance metrics
│       └── test_performance.csv             # Test set metrics
│
├── environment.yml                          # Conda environment specification
├── requirements.txt                         # pip dependencies
└── README.md                                # This file


---

## 🚀 Setup & Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate consumer-staples-forecast
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Run scripts in this exact order:Step 1: Download Data

cd src
python data_loader.py

cd src
python prepreocessing.py

cd src
python create_train_validation_test_split.py

cd src
python model_evaluate.py

cd src
python train_all.py

cd src
python test_all

cd src
python feature_importance.py

cd src
python backtesting.py

cd src
python monte_carlo.py