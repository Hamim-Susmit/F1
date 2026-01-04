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
│  - 150+ features per driver per race            │
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
