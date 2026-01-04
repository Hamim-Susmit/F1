import os

from f1_prediction_pipeline import F1RacePredictor


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")
    predictor = F1RacePredictor(database_url=database_url)
    predictions = predictor.predict_race(202601)["predictions"]
    print(f"Predicted winner: {predictions[0]['driver']}")


if __name__ == "__main__":
    main()
