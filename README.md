# VIX Direction Prediction with Deep Learning 🧠⚡

This project forecasts the **daily directional movement of the VIX (Volatility Index)** using an LSTM neural network trained on engineered features from SPY, QQQ, and VIX data.

It features a full pipeline for:
- 🧹 Data loading and feature engineering
- 🪟 Rolling-window time series cross-validation
- 🧠 LSTM model training and evaluation
- 📊 Interactive model performance dashboard via Streamlit
- 🔁 Versioned result tracking and comparison

---

## 💼 Motivation

Volatility forecasting plays a crucial role in trading, hedging, and risk management. This project demonstrates:
- Deep learning applied to financial time series
- Robust validation under realistic constraints (no future leakage)
- Evaluation beyond accuracy: class imbalance handling, baseline comparisons, visual diagnostics

Ideal for showcasing applied machine learning + finance skills to employers or collaborators.

---

## 🔧 Project Structure

```
.
├── data/                  # Data loading and feature engineering
│   ├── raw.py
│   └── loader.py
├── models/                # Model, dataset, and splitter definitions
│   ├── lstm.py
│   ├── dataset.py
│   └── splitter.py
├── src/pipeline/          # Data and training pipeline orchestration
│   ├── data_pipeline.py
│   └── training_pipeline.py
├── config/                # Config presets (quick_test, dev, prod, etc.)
│   └── settings.py
├── main.py                # Main entrypoint for running the pipeline
├── vix_dashboard.py       # Streamlit dashboard for visualizing results
├── notebooks/             # (Optional) EDA notebooks
└── results/               # Auto-saved JSON results, plots, and summaries
```

---

## 🚀 How to Run

### 1. 📦 Install Requirements

```bash
pip install -r requirements.txt
```

(Or use `conda` or `venv` as preferred.)

---

### 2. 🧪 Run a Quick Test (for speed/debugging)

```bash
python main.py --test
```

### 3. 🏭 Run the Full Production Pipeline

```bash
python main.py --prod
```

You can also run:

```bash
python main.py --config development
python main.py --config heavy_model
python main.py --config light_model
```

This generates:
- `results/pipeline_config_<config>_vX.json`
- `results/lstm_training_<config>_vX.json`
- `results/complete_pipeline_<config>_vX.json`
- `results/lstm_training_<config>_vX.png`

---

### 4. 📊 Launch the Interactive Dashboard

```bash
streamlit run vix_dashboard.py
```

Explore:
- Model comparison across configurations
- Fold-by-fold metrics
- Visual diagnostics and PNG charts
- Raw JSON results for each run

---

## 📉 Features Engineered

The model uses:
- Daily returns of SPY, QQQ, and VIX
- 5-day rolling volatilities
- QQQ–SPY return spread
- Discrete directional target for VIX (`Down`, `Neutral`, `Up`) based on ±1% thresholds

---

## 🧠 Model Details

- PyTorch LSTM with 2 layers, dropout, and MLP classifier head
- Class imbalance handled via weighted loss
- Rolling window splitter for realistic cross-validation
- Model performance compared to a 33.3% random baseline
- Full analysis includes accuracy, confusion matrix, loss trends, and fold stats

---

## 📁 Example Results

```
Config        | Version | Accuracy   | Improvement  | Date
---------------------------------------------------------------
production    | v4      | 48.5%      | +45.7%       | 2025-07-27
light_model   | v2      | 42.1%      | +26.3%       | 2025-07-25
heavy_model   | v1      | 51.3%      | +54.1%       | 2025-07-24
```

Check `/results` folder for full JSON logs and PNG plots.

---

## ✅ Requirements

- Python 3.9+
- yfinance
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- torch
- streamlit
- plotly

Install with:

```bash
pip install -r requirements.txt
```

---

## 📘 License

MIT License — feel free to fork, learn, adapt, and use.

---

## 💡 Future Ideas

- Add GARCH or Transformer baselines
- Integrate backtesting and trading signal simulation
- Expand to other volatility indexes (e.g., VXN, VXO)
