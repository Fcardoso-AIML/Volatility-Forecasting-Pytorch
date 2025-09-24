# VIX Direction Prediction with Deep Learning

**Developed by Francisco Cardoso**  
**MSc in Mathematics Economics, University of Copenhagen**

This project forecasts the **daily directional movement of the VIX (Volatility Index)** using an **LSTM neural network with a multi-layer perceptron (MLP) classifier head**, trained on engineered features from SPY, QQQ, and VIX data.

It features a full pipeline for:
* Data loading and feature engineering
* Rolling-window time series cross-validation
* LSTM + MLP classifier training and evaluation
* Interactive model performance dashboard via Streamlit
* Versioned result tracking and comparison

## Motivation

Volatility forecasting plays a crucial role in trading, hedging, and risk management. This project demonstrates:
* Deep learning applied to financial time series
* Robust validation under realistic constraints (no future leakage)
* Evaluation beyond accuracy: class imbalance handling, baseline comparisons, visual diagnostics

Ideal for showcasing applied machine learning + finance skills to employers or collaborators.

## Project Structure

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

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Run a Quick Test (for speed/debugging)

```bash
python main.py --test
```

### 3. Run the Full Production Pipeline

```bash
python main.py --prod
```

Or use a preset:

```bash
python main.py --config development
python main.py --config heavy_model
python main.py --config light_model
```

Generates:
* `results/pipeline_config_<config>_vX.json`
* `results/lstm_training_<config>_vX.json`
* `results/complete_pipeline_<config>_vX.json`
* `results/lstm_training_<config>_vX.png`

### 4. Launch the Interactive Dashboard

```bash
python -m streamlit run vix_dashboard.py
```

Explore:
* Model comparison across configurations
* Fold-by-fold metrics
* Visual diagnostics and PNG charts
* Raw JSON results for each run

## Dashboard Screenshots

The interactive dashboard provides comprehensive model analysis:

**Individual Model Analysis**
![Individual Model Analysis](C:\Users\Francisco Cardoso\OneDrive\Ambiente de Trabalho\Projects\Volatility Forecasting\screenshots\Individual_Model_Analysis.png)

**Model Comparison View** 
![Model Comparison](C:\Users\Francisco Cardoso\OneDrive\Ambiente de Trabalho\Projects\Volatility Forecasting\screenshots\Model_Comparision.png)

**Model Visualizations**
![Model Visualizations](C:\Users\Francisco Cardoso\OneDrive\Ambiente de Trabalho\Projects\Volatility Forecasting\screenshots\Model_Visualizations.png)

**Performance Overview**
![Performance Overview](C:\Users\Francisco Cardoso\OneDrive\Ambiente de Trabalho\Projects\Volatility Forecasting\screenshots\Overview.png)

## Features Engineered

The model uses:
* Daily returns of SPY, QQQ, and VIX
* 5-day rolling volatilities
* QQQ–SPY return spread
* Discrete directional target for VIX (`Down`, `Neutral`, `Up`) based on ±1% thresholds

## Model Details

* **Backbone:** PyTorch LSTM with 2 stacked layers (hidden size 64) + dropout
* **Head:** Multi-layer perceptron (MLP) classifier
   * Hidden layers: 64 → 32 → 16 → 3
   * Activations: ReLU + Dropout
   * Final layer outputs logits for 3 classes (Down, Neutral, Up)
* **Training:** Class imbalance handled via weighted loss
* **Validation:** Rolling window splitter for realistic cross-validation
* **Evaluation:** Compared against a 33.3% random baseline
* **Analysis:** Accuracy, confusion matrix, loss trends, fold stats

## Example Results

| Config        | Version | Accuracy | Improvement | Date       |
|---------------|---------|----------|-------------|------------|
| production    | v4      | 48.5%    | +45.7%      | 2025-07-27 |
| light_model   | v2      | 42.1%    | +26.3%      | 2025-07-25 |
| heavy_model   | v1      | 51.3%    | +54.1%      | 2025-07-24 |

See `/results` for JSON logs and PNG plots.

## Requirements

* Python 3.9+
* yfinance, pandas, numpy
* scikit-learn
* matplotlib, seaborn
* torch
* streamlit, plotly

## License

MIT License — feel free to fork, learn, adapt, and use.