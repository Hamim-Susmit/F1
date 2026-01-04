from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from f1_prediction_pipeline import F1RacePredictor, ModelEvaluator, ScenarioSimulator


def get_predictor() -> F1RacePredictor:
    database_url = os.environ.get("DATABASE_URL")
    model_dir = os.environ.get("MODEL_DIR", "models")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")
    return F1RacePredictor(database_url=database_url, model_dir=model_dir)


def main() -> None:
    st.set_page_config(page_title="F1 Predictions 2026", layout="wide")
    st.sidebar.title("F1 2026 Predictions")

    race_name = st.sidebar.text_input("Race Name", "Australian GP")
    race_id = st.sidebar.number_input("Race ID", min_value=1, value=202601, step=1)
    scenario = st.sidebar.radio("Weather Scenario", ["Base Forecast", "Dry", "Wet", "Mixed"])

    st.title(f"🏎️ {race_name} Predictions")

    predictor = get_predictor()
    if scenario != "Base Forecast":
        predictor.set_weather_scenario(scenario.lower())

    prediction_payload = predictor.predict_race(race_id=race_id, use_latest_data=True)
    predictions = prediction_payload["predictions"]

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        st.subheader("Predicted Results")
        for i, pred in enumerate(predictions[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.write(
                f"{medal} **{pred['driver']}** ({pred['team']}) - "
                f"{pred['confidence']:.0f}% confidence"
            )

    with col2:
        st.subheader("Key Factors")
        st.write("Circuit characteristics, weather impact, and championship context.")

    with col3:
        st.subheader("Probabilities")
        for pred in predictions[:5]:
            st.write(
                f"{pred['driver']}: Podium {pred['podium_probability']:.0%}, "
                f"DNF {pred['dnf_probability']:.0%}"
            )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[p["driver"] for p in predictions[:10]],
            y=[p["confidence"] for p in predictions[:10]],
            marker_color="lightblue",
        )
    )
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Driver",
        yaxis_title="Confidence %",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Why These Predictions?")
    st.write("Use SHAP summary artifacts from model training for deeper explanation.")

    simulator = ScenarioSimulator(predictor)
    if st.checkbox("Compare Weather Scenarios"):
        scenario_results = simulator.simulate_weather_scenarios(race_id)
        st.json(scenario_results)

    st.subheader("Model Performance This Season")
    evaluator = ModelEvaluator()
    metrics = evaluator.generate_report(season=2026)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Winner Accuracy", f"{metrics['winner_accuracy']:.1%}")
    col2.metric("Top 3 Accuracy", f"{metrics['top3_accuracy']:.1%}")
    col3.metric("Avg Position Error", f"{metrics['avg_position_mae']:.2f}")
    col4.metric("Races Predicted", metrics["total_races"])


if __name__ == "__main__":
    main()
