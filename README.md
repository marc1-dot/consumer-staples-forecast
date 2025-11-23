# Consumer Staples Forecasting

## 🧠 Project Overview
This project forecasts **revenues and stock prices** for major companies in the **consumer staples sector** — specifically **Nestlé**, **Procter & Gamble**, **Unilever**, **Coca-Cola**, and **PepsiCo** — using **machine learning** and **econometric modeling** techniques.

The goal is to compare different models (Linear Regression, Neural Networks, Random Forests, and Gradient Boosting) to determine which provides the most accurate and robust forecasts over time.

---

## 🎯 Objectives
1. **Collect** and clean financial and market data automatically using Python APIs (Yahoo Finance, FinancialModelingPrep).
2. **Analyze** relationships between financial fundamentals and stock prices.
3. **Train** multiple models on historical data (2015–2022).
4. **Test and backtest** models on unseen data (2023–2025).
5. **Evaluate** model performance using statistical and predictive metrics.
6. **Interpret** and visualize forecasting results.

---

## 🧰 Technologies Used
- **Python 3.10+**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for ML models
- `yfinance` and `financialmodelingprep` for data collection
- `xgboost` for advanced gradient boosting
- `statsmodels` for econometric analysis

---

## 📁 Repository Structure
```
consumer-staples-forecast/
├── README.md                # Project overview
├── proposal.md              # Project proposal
├── requirements.txt         # Dependencies
├── src/                     # Source code
│   ├── data_loader.py       # Data collection
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   ├── models.py            # ML model definitions
│   ├── evaluation.py        # Metrics & visualization
│   └── main.py              # Main script
├── tests/                   # Unit & integration tests
├── results/                 # Model results and figures
├── docs/                    # Documentation (architecture, notes)
└── AI_USAGE.md              # Disclosure of AI assistance
```

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/consumer-staples-forecast.git
cd consumer-staples-forecast
pip install -r requirements.txt
```

---

## ▶️ Usage
Run the full forecasting pipeline:
```bash
python src/main.py
```
Or open the notebook for interactive exploration:
```bash
jupyter notebook notebooks/forecasting_analysis.ipynb
```

---

## 📊 Evaluation Metrics
Models will be evaluated using:
- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

---

## 🧩 Roadmap
- [x] Create project structure
- [x] Define project proposal
- [ ] Automate data collection
- [ ] Develop feature engineering pipeline
- [ ] Train and evaluate ML models
- [ ] Write final report and presentation

---

## 📜 License & Academic Integrity
This project was created as part of the **HEC Lausanne - Advanced Programming (Fall 2025)** course.  
AI tools were used for assistance and code generation in accordance with the course’s AI usage policy.

---

**Author:** Marc Birchler  
**Supervisor:** Prof. [Instructor Name]  
**Date:** November 2025
