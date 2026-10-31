<div align="center">

# ECOPRICE
### A Real-Time Dynamic Pricing and Spoilage Prediction Algorithm

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB8E24?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![RL](https://img.shields.io/badge/Reinforcement%20Learning-PPO%20%7C%20Gymnasium-orange?style=for-the-badge)](https://gymnasium.farama.org/)

<br/>

**A reinforcement learning and supervised demand forecasting platform designed to minimize perishable wastage and maximize retail profits.**

[Architecture](#system-architecture) &bull; [Quick Start](#getting-started)

</div>

---

## About

Perishable grocery departments (dairy, fresh produce, meat, poultry, seafood, and bakery) operate under very thin margins and short shelf-life limits. Traditional retail supermarkets apply static pricing until goods reach their terminal sell-by dates, causing wastage of goods and losses over unsold inventory.

**EcoPrice** replaces static pricing with a data-driven pricing intelligence engine. It couples:
1. **Supervised Demand Forecasting (XGBoost):** Anticipates daily item demand using 50 engineered features across lag sales, rolling statistical trends, calendar periodicity, upcoming regional festivals, and ambient meteorological conditions.
2. **Arrhenius Spoilage Decay Modeling:** Adjusts real-time product deterioration speed according to ambient temperature and relative humidity across Indian agro-climatic zones.
3. **Multi-Objective Reinforcement Learning (PPO):** Balances perishable inventory salvage value against waste write-off penalties ($\lambda = 0.75$) and brand discount penalties ($\gamma = 0.35$) inside a custom Gymnasium environment.
4. **Store Manager Decision Dashboard:** Provides store staff with real-time shrinkage telemetry, color-coded shelf expiration countdowns, and one-click algorithmic markdown approvals.

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
├── app.py                                
├── start_live.py                         
├── start_ecoprice.bat                    
├── requirements.txt                      
├── Dockerfile                            
├── Procfile / render.yaml                
│
├── src/                                  
│   ├── model_service.py                  
│   ├── perishable_classifier.py          
│   ├── weather_service.py               
│   ├── preprocess.py                    
│   ├── feature_extraction.py             
│   └── email_service.py                  
│
├── models/                               
│   ├── environment.py                    
│   ├── ppo_agent.py                      
│   ├── xgboost_model.py                  
│   ├── xgboost_causal.py                                       
│
├── templates/                           
│   ├── base_app.html                     
│   ├── dashboard.html                    
│   ├── expiry.html                       
│   ├── ppo.html                          
│   ├── forecast.html                     
│   ├── sales.html                        
│   └── settings.html                     
│
├── static/                               
│   ├── css/                              
│   ├── js/                               
│   ├── images/                           
│   └── audio/                            
│
├── data/                                 
│   ├── sales_register_data_golden_backup.csv 
│   ├── kerala_clean.csv                  
│   ├── inventory_state.json              
│   └── settings.json                     
│
└── documentation/                        
    ├── Phase_1_Preprocessing.docx        
    ├── Phase_2_Feature_Extraction.docx   
    ├── Phase_3_XGBoost_Training.docx     
    ├── Phase_4_PPO_Training.docx         
    
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
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
