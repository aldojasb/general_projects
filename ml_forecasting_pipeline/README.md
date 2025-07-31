# **User Story**

“As a business analyst, I want to click a button when I suspect our model is underperforming, so a new version is retrained using recent data and safely deployed to production.”



## **Flow Overview**



```tex
[UI Button Click] 
     ↓
[FastAPI Backend Trigger]
     ↓
[Trigger DAG via Airflow API]
     ↓
[Airflow DAG Tasks]
   ├── Fetch new data from API
   ├── Train new model
   ├── Evaluate new model
   └── Deploy if better
```



## **Updated and Clearer Project Structure**



```tex
ml_forecasting_pipeline/
│
├── backend/                         ← FastAPI backend
│   ├── api/                         ← API routes
│   │   └── retrain.py               ← POST /retrain
│   ├── services/                    ← Business orchestration logic
│   │   └── airflow_trigger.py       ← Triggers Airflow DAG
│   ├── config.py                    ← Settings (Airflow URL, DAG name, env vars)
│   └── main.py                      ← FastAPI app entrypoint
│
├── dags/                            ← Airflow DAGs
│   └── retrain_forecasting_model.py← Full pipeline logic
│
├── ml/                              ← Model logic (can be imported in DAG)
│   ├── training.py                  ← Training logic
│   ├── evaluation.py               ← Evaluation metrics (e.g., MAE, MAPE)
│   └── deployment.py               ← Deployment logic (save model, update endpoint)
│
├── ui/                              ← Frontend UI (minimal demo)
│   └── index.html                   ← Button: "Retrain Model"
│
├── docker-compose.yml              ← Full stack orchestration
├── requirements.txt                ← Python deps (FastAPI, requests, etc.)
└── README.md                       ← How it works + usage instructions
```





## **Building Blocks **

### **Step 1: Initialize the Repo**

- Name: ml_forecasting_pipeline
- Setup folders and .gitignore, README
- Add pyproject.toml or requirements.txt

> **Goal:** Establish a modular and testable layout (SRP)

------

### **Step 2: FastAPI Backend to Trigger Retraining**

- POST /retrain route
- Abstract logic into airflow_trigger.py
- Configs in config.py

> **SOLID:**

- **SRP**: Each module does one thing (routing, config, logic)
- **DIP**: API route depends on IAirflowTrigger, not concrete details

------

### **Step 3: Airflow DAG Definition**

**Pipeline steps:**

1. Fetch data from a public API (e.g., Open-Meteo, Alpha Vantage)
2. Clean + save to file
3. Train new forecasting model (e.g., Prophet, XGBoost)
4. Evaluate new vs. old
5. If better → replace model in prod_model.pkl

> **SOLID:**

- **OCP**: You can add “Notify via Slack” or “Retrain multiple models” as new steps
- **LSP**: New ML pipelines can reuse DAG pattern

------

### **Step 4: ML Logic Modules (ml/)**

- training.py: Load data, fit model, save artifact
- evaluation.py: Compare metrics (e.g., MAE < threshold or MAE < old)
- deployment.py: Replace prod_model.pkl, update S3, or reload model server

> **SOLID:**

- **SRP**: Training ≠ Evaluation ≠ Deployment
- **ISP**: Each file is a focused interface
- **DIP**: DAG tasks use evaluate(model1, model2) without knowing internals

------

### **Step 5: UI Trigger**

- HTML page with one button
- Uses fetch('/retrain', { method: 'POST' })

> **SOLID:**

- Keeps frontend dumb (SRP)
- Backend owns the business logic

------

### **Step 6: Docker Compose**

Services:

- fastapi_backend
- airflow_webserver, airflow_scheduler
- airflow_postgres, airflow_worker
- (optional) mlflow_server, redis

> **Goal:** Run full stack locally with one command

------

### **Step 7: Testing + Logging**

- Add a test: “Click button → retrain pipeline runs”
- Logs stored per task in Airflow
- Test fallback path: model rejected if not better

------

### **Step 8: Document It**

- Clear use case: “Business-driven model retraining”
- Show architecture diagram
- Demo: screenshot or Loom video of button + pipeline running



------



## **Suggested Project Evolution Path**



| **Phase** | **Structure**                                        | **Benefit**                        |
| --------- | ---------------------------------------------------- | ---------------------------------- |
| Phase 1   | Keep current mono-repo + modular folders             | Fast to build, shows orchestration |
| Phase 2   | Dockerize each ML module                             | Prepares for service isolation     |
| Phase 3   | Replace Airflow PythonOperator → DockerOperator      | True separation of services        |
| Phase 4   | Deploy everything to Kubernetes (Airflow + services) |                                    |