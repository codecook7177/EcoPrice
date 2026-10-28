<div align="center">

# ECOPRICE
### A Real-Time Dynamic Pricing and Spoilage Prediction Algorithm

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB8E24?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![RL](https://img.shields.io/badge/Reinforcement%20Learning-PPO%20%7C%20Gymnasium-orange?style=for-the-badge)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

**A multi-objective reinforcement learning and supervised demand forecasting platform designed to minimize physical grocery shrinkage, optimize perishable inventory markdowns, and protect store margins.**

[Explore Features](#key-features) &bull; [Architecture](#system-architecture) &bull; [Empirical Results](#empirical-results) &bull; [Quick Start](#getting-started) &bull; [Thesis Documentation](#documentation)

</div>

---

## Executive Summary

Perishable grocery departments (dairy, fresh produce, meat, poultry, seafood, and bakery) operate under razor-thin margins and stringent shelf-life limits. Traditional retail supermarkets apply static, uniform pricing until goods reach their terminal sell-by dates, causing steep markdowns at the last minute or complete inventory write-offs.

**EcoPrice** replaces static pricing with a closed-loop, data-driven pricing intelligence engine. It couples:
1. **Supervised Demand Forecasting (XGBoost):** Anticipates daily item demand using 50 engineered features across lag sales, rolling statistical trends, calendar periodicity, upcoming regional festivals, and ambient meteorological conditions.
2. **Arrhenius Spoilage Decay Modeling:** Adjusts real-time product deterioration speed according to ambient temperature and relative humidity across Indian agro-climatic zones.
3. **Multi-Objective Reinforcement Learning (PPO):** Balances perishable inventory salvage value against waste write-off penalties ($\lambda = 0.75$) and brand discount penalties ($\gamma = 0.35$) inside a custom Gymnasium environment.
4. **Store Manager Decision Dashboard:** Provides store staff with real-time shrinkage telemetry, color-coded shelf expiration countdowns, and one-click algorithmic markdown approvals.

---

## User Interface & System Walkthrough

<div align="center">

### Store Management Dashboard
![EcoPrice Dashboard](documentation/thesis_screens/screen_03_dashboard.png)
*Executive overview displaying real-time shrinkage rate, estimated spoilage loss, revenue saved, active perishable batches, and quick action controls.*

</div>

### Interface Capabilities

| Screen | View | Description |
| :--- | :--- | :--- |
| **Expiry Tracker** | ![Expiry Tracker](documentation/thesis_screens/screen_04_expiry_tracker.png) | Color-coded inventory shelf-life ledger with visual urgency alerts (Green: $\ge$ 4 days, Amber: 2–3 days, Red: 1 day). |
| **Dynamic Pricing Engine** | ![Dynamic Pricing](documentation/thesis_screens/screen_05_dynamic_pricing.png) | PPO algorithmic markdown approval panel showing recommended discounts (0%–30%), base retail prices, and sales lift. |
| **Demand Forecasting** | ![Demand Forecast](documentation/thesis_screens/screen_06_demand_forecast.png) | Departmental daily sales predictions compared against 7-day historical moving averages with weather impact overlays. |
| **Store Settings & GPS** | ![Settings](documentation/thesis_screens/screen_08_settings_location.png) | GPS auto-detection of store state/district for climate syncing, manager profile avatar uploads (<3MB), and algorithm weights. |

---

## Key Features

- **50-Feature Chronological ML Pipeline:** Extracts temporal cycles, rolling window statistical aggregations (7d and 14d mean, standard deviation, min, max), calendar lag intervals, holiday indicators, and climate interaction metrics.
- **Microclimate-Aware Spoilage Acceleration:** Uses Arrhenius-derived kinetic multipliers based on local temperature ($T$) and humidity ($RH$) to calibrate daily shelf decay.
- **PPO Reinforcement Policy:** 200,000 timestep training in a Gymnasium retail simulation environment delivering an **85.7% reduction in perishable spoilage waste**.
- **Human-in-the-Loop Governance:** Algorithmic suggestions require store manager verification with options for individual approval, bulk execution, manual adjustment, or instant undo.
- **Dynamic Catalog Management:** Real-time intake for incoming delivery batches with automated perishable classification heuristics and live inventory reconciliation.
- **Factory Baseline Reset:** One-click rollback restores inventory counts, datasets, and feature matrices to pristine factory baseline records.
- **Audit Logging & CSV Export:** Download complete historical logs of manager pricing decisions for store accounting audits.

---

## System Architecture

```
                       [ PHYSICAL SUPERMARKET ENVIRONMENT ]
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  Point-of-Sale (POS)          Ambient Temperature          Regional Festival
   Daily Sales Stream            & Humidity Feeds                Calendar
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                       [ 50-FEATURE EXTRACTION PIPELINE ]
                                        │
                                        ▼
                       [ XGBOOST DEMAND FORECASTER ]
                           Predicts Expected Demand
                                        │
                                        ▼
                    [ GYMNASIUM RETAIL SIMULATION ENV ]
                    State: (Stock, Days Left, Climate, Demand)
                                        │
                                        ▼
                        [ PPO REINFORCEMENT LEARNING ]
                     Optimizes Discount Action (0% - 30%)
                                        │
                                        ▼
                       [ STORE MANAGER WEB TERMINAL ]
                    Review -> Approve / Override / Sync
```

---

## Empirical Results

The demand forecasting and reinforcement learning architectures were evaluated using unseen chronological test splits and retail simulation benchmarks.

### 1. Demand Forecasting Performance (XGBoost)

| Product Department | Samples | $R^2$ Score | MAE (Units/Day) | RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **Dairy & Eggs** | 2,840 | **0.8912** | 16.45 | 22.18 |
| **Fruits & Vegetables** | 4,215 | **0.8741** | 19.82 | 26.54 |
| **Meat, Poultry & Seafood** | 1,980 | **0.8490** | 23.10 | 31.42 |
| **Bakery & Confectionery** | 1,545 | **0.8320** | 25.11 | 34.05 |
| **Combined Perishable Catalog** | **10,580** | **0.8653** | **21.12** | **28.55** |

### 2. Reinforcement Learning Policy Evaluation (PPO vs Baselines)

| Performance Metric | Static Pricing (No Discount) | Heuristic Rule-Based | EcoPrice PPO Agent | Relative Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Inventory Waste Rate** | 21.40% of inventory | 9.85% of inventory | **3.06% of inventory** | **-85.7% Waste Reduction** |
| **Salvage Revenue Recovery** | ₹14,200 | ₹19,850 | **₹24,850** | **+75.0% Revenue Lift** |
| **Gross Margin Retention** | 78.60% | 83.15% | **89.42%** | **+10.82% Margin Preservation** |
| **Average Markdown Rate** | 0.00% (or write-off) | 22.50% | **14.20%** | Controlled Price Integrity |

---

## Technology Stack

- **Backend:** Python 3.10+, Flask 3.0, Werkzeug, Gunicorn
- **Machine Learning & RL:** XGBoost, Scikit-Learn, Gymnasium, Stable-Baselines3 (PPO), PyTorch, NumPy, Pandas, Joblib
- **Frontend:** Jinja2 Templates, Vanilla JavaScript (ES6+), CSS3 (Custom Responsive Layouts), Chart.js 4.4
- **Edge Deployment & Networking:** Cloudflare Tunnels (`cloudflared`), Docker
- **Data Formats & Storage:** JSON State Ledgers, CSV POS Archives

---

## Directory Structure

```text
ecoprice/
├── app.py                                # Primary Flask server and API router
├── start_live.py                         # One-click launcher with Cloudflare Tunnel
├── start_ecoprice.bat                    # Windows batch launcher
├── requirements.txt                      # Production dependencies
├── Dockerfile                            # Docker container configuration
├── Procfile / render.yaml                # Cloud hosting specifications
├── EcoPrice_Capstone_Project_Thesis.docx # University Capstone Thesis (58 pages)
│
├── src/                                  # Core business and ML services
│   ├── model_service.py                  # Catalog coordinator and model loader
│   ├── perishable_classifier.py          # Perishable gatekeeper heuristics
│   ├── weather_service.py                # Meteorological APIs and Arrhenius index
│   ├── preprocess.py                     # Raw dataset sanitization pipeline
│   ├── feature_extraction.py             # 50-feature chronological pipeline
│   └── email_service.py                  # Transactional notification service
│
├── models/                               # Model code and weights
│   ├── environment.py                    # Custom Gymnasium retail environment
│   ├── ppo_agent.py                      # PPO training loop and agent checkpointing
│   ├── xgboost_model.py                  # Demand forecasting training pipeline
│   ├── xgboost_causal.py                 # Feature ablation model
│   └── saved/                            # Model binary weights and evaluation plots
│       ├── xgboost_demand_kerala.joblib  # Trained XGBoost forecaster
│       ├── ppo_discount_kerala.zip       # Trained PPO actor-critic policy
│       └── plots/                        # Training curves and feature importance plots
│
├── templates/                            # Frontend Jinja2 HTML templates
│   ├── base_app.html                     # Main application layout with top navigation
│   ├── dashboard.html                    # Store executive dashboard
│   ├── expiry.html                       # Perishable expiry tracker ledger
│   ├── ppo.html                          # Algorithmic pricing approval terminal
│   ├── forecast.html                     # Demand forecasting curves and analytics
│   ├── sales.html                        # POS daily sales register
│   └── settings.html                     # Store location, GPS, avatar, and tuning
│
├── static/                               # Static web assets
│   ├── css/                              # public.css, pages.css, style.css
│   ├── js/                               # main.js, ambient.js
│   ├── images/                           # hero-art.png
│   └── audio/                            # ambient.mp3 (background grocery audio)
│
├── data/                                 # Datasets and schemas
│   ├── sales_register_data_golden_backup.csv # Pristine factory sales records
│   ├── kerala_clean.csv                  # Preprocessed grocery transactions
│   ├── inventory_state.json              # Active stock counts across 38 SKUs
│   └── settings.json                     # Store parameters and shelf lives
│
└── documentation/                        # Project documentation
    ├── Phase_1_Preprocessing.docx        # Phase 1 documentation
    ├── Phase_2_Feature_Extraction.docx   # Phase 2 documentation
    ├── Phase_3_XGBoost_Training.docx     # Phase 3 documentation
    ├── Phase_4_PPO_Training.docx         # Phase 4 documentation
    ├── EcoPrice_Master_Review_Preparation_Guide.docx # Viva preparation guide
    └── thesis_screens/                   # 8 high-resolution UI walkthrough images
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/ecoprice.git
cd ecoprice
```

### 3. Set Up Virtual Environment
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application

#### Standard Local Execution:
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

#### One-Click Live Cloud Demo (with Cloudflare Tunnel):
```bash
python start_live.py
```
This automatically initiates the local Flask server, resolves local Wi-Fi LAN access, and establishes a secure public HTTPS Cloudflare tunnel for external device access.

---

## API Documentation

| Endpoint | Method | Access | Purpose |
| :--- | :---: | :---: | :--- |
| `/api/suggestion/<id>/<action>` | `POST` | Protected | Approve (`accept`), reject (`reject`), or revert (`undo`) a discount recommendation. |
| `/api/bulk-action` | `POST` | Protected | Apply or dismiss all pending PPO recommendations in batch. |
| `/api/update-stock` | `POST` | Protected | Modify on-hand unit count for an individual perishable SKU. |
| `/api/add-product` | `POST` | Protected | Intake a new delivery batch with perishable classification validation. |
| `/api/delete-product` | `POST` | Protected | Remove an active SKU from store monitoring. |
| `/api/detect-location` | `POST` | Protected | Ingests GPS coordinates, reverse geocodes city, and syncs climate multipliers. |
| `/api/upload-avatar` | `POST` | Protected | Validates (<3MB) and stores manager profile photos. |
| `/api/export-decision-history` | `GET` | Protected | Generates and downloads an accounting CSV audit log of all pricing decisions. |
| `/api/reset-dataset` | `POST` | Protected | Restores database, inventory caches, and settings to golden factory baseline. |

---

## Documentation

Full academic reports and review guides are provided inside the `documentation/` folder:
- **Comprehensive Project Thesis:** [EcoPrice_Capstone_Project_Thesis.docx](EcoPrice_Capstone_Project_Thesis.docx) (58 Pages, 12 Figures, 10 Tables)
- **Phase Reports:**
  - [Phase 1: Dataset Preprocessing & Cleaning](documentation/Phase_1_Preprocessing.docx)
  - [Phase 2: 50-Feature Engineering Specification](documentation/Phase_2_Feature_Extraction.docx)
  - [Phase 3: Supervised XGBoost Training](documentation/Phase_3_XGBoost_Training.docx)
  - [Phase 4: Gymnasium & PPO Reinforcement Learning](documentation/Phase_4_PPO_Training.docx)
- **Viva & Defense Guide:** [EcoPrice_Master_Review_Preparation_Guide.docx](documentation/EcoPrice_Master_Review_Preparation_Guide.docx)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
