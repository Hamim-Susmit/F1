# F1 2026 Prediction System - Technical Documentation

## System Architecture
```
┌─────────────────────────────────────────────────┐
│          Data Collection Layer                  │
│  - FastF1 API                                   │
│  - Weather APIs                                 │
│  - Manual data entry                            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Data Storage (PostgreSQL)              │
│  - Historical races (2010-2025)                 │
│  - Weather data                                 │
│  - Circuit characteristics                      │
│  - Driver/team metadata                         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│       Feature Engineering Pipeline              │
│  - Dozens of engineered features per driver     │
│  - Real-time updates from practice/quali        │
│  - Interaction features                         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Model Ensemble                         │
│  - XGBoost (35%)                                │
│  - LightGBM (30%)                               │
│  - Random Forest (25%)                          │
│  - Neural Network (10%)                         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│          Prediction API (FastAPI)               │
│  - REST endpoints                               │
│  - WebSocket for live updates                   │
│  - Caching layer (Redis)                        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│       Frontend Dashboard (Streamlit)            │
│  - Interactive predictions                      │
│  - Scenario analysis                            │
│  - Model performance tracking                   │
└─────────────────────────────────────────────────┘
```

## Key Features
- Dozens of engineered features per prediction (driver, team, and situational signals)
- Ensemble of 4 models for robustness
- Real-time updates from practice and qualifying
- Scenario simulation (weather, grid penalties, etc.)
- Explainable AI (SHAP values explain each prediction)
- Continuous learning (retrain after each race)
- 2026-specific adaptations for new regulations

## Performance Targets
| Metric | Target | Current |
| --- | --- | --- |
| Winner Accuracy | 45% | TBD |
| Top 3 Accuracy | 65% | TBD |
| Top 10 Accuracy | 80% | TBD |
| Position MAE | <2.5 | TBD |
| API Latency p95 | <500ms | TBD |
