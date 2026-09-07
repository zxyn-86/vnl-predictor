
import preprocessing
import features 
import model
import profiles

from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_classifier_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


def print_banner(title):
    border = "+" + "=" * 52 + "+"
    print(border)
    print(f"| {title:^50} |")
    print(border)


def probability_bar(probability, width=30):
    filled = round(probability * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_feature_comparison(team1, team2, matchup_features):
    print("\nFeature advantages (positive values favor Team 1)")
    print("-" * 62)
    print(f"{'Feature':<30} {'Team 1 lead':>11} {'Team 2 lead':>11} {'Diff':>7}")
    print("-" * 62)

    for column in features.FeatureEngineer.FEATURE_COLUMNS:
        feature_name = column.removesuffix("_Diff")
        difference = float(matchup_features[column].iloc[0])
        team1_value = difference if difference >= 0 else 0
        team2_value = -difference if difference < 0 else 0
        print(
            f"{feature_name:<30} {team1_value:>11.3f} "
            f"{team2_value:>11.3f} {difference:>7.3f}"
        )


def main():


    teams = [
    "Argentina",
    "Belgium",
    "Brazil",
    "Bulgaria",
    "Canada",
    "China",
    "Cuba",
    "France",
    "Germany",
    "Iran",
    "Italy",
    "Japan",
    "Poland",
    "Serbia",
    "Slovenia",
    "Türkiye",
    "Ukraine",
    "United States",
]
    print("Welcome to vballPredictor!")
    print("This tool predicts the winner of a volleyball match based on historical data.")
    print("Please choose an option:")
    print("1. Predict a matchup")
    print("2. Run historical data processing and model training")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        print("Available teams:")
        for i, team in enumerate(teams, start=1):
            print(f"{i}. {team}")

        
        while True:
            team1 = input("Enter the name of Team 1: ")
            if team1 in teams:
                break
            print(f"Team {team1} is not in the list of available teams. Try again.")

        while True:
            team2 = input("Enter the name of Team 2: ")
            if team2 in teams:
                break
            print(f"Team {team2} is not in the list of available teams. Try again.")

        matchup(team1, team2)
    elif choice == "2":
        historical()
    else:
        print("Invalid choice. Please enter 1 or 2.")


def matchup(team1, team2):
    print_banner(f"{team1} vs {team2}")
    print("Building matchup from recent team history...\n")

    df = features.FeatureEngineer().build_features()
    matchup_features = profiles.get_matchup_features(df, team1, team2)

    scaler = joblib.load(SCALER_PATH)
    saved_model = joblib.load(MODEL_PATH)

    scaled_features = scaler.transform(
        matchup_features[features.FeatureEngineer.FEATURE_COLUMNS]
    )

    prediction = saved_model.predict(scaled_features)[0]
    decision_score = saved_model.decision_function(scaled_features)[0]

    # This is a confidence-like score, not a calibrated probability.
    team1_confidence = 1 / (1 + np.exp(-decision_score))
    team2_confidence = 1 - team1_confidence

    winner = team1 if prediction == 1 else team2

    print(f"Predicted winner: {winner}")
    print(f"\n{team1:<18} {probability_bar(team1_confidence)} {team1_confidence:>6.1%}")
    print(f"{team2:<18} {probability_bar(team2_confidence)} {team2_confidence:>6.1%}")
    print(f"\nModel decision score: {decision_score:.3f}")
    print_feature_comparison(team1, team2, matchup_features)

    return {
        "winner": winner,
        f"{team1}_chance": team1_confidence,
        f"{team2}_chance": team2_confidence,
        "decision_score": decision_score,
    }



def historical():
    print("Hello from vballPredictor!")
    preprocessor = preprocessing.DataPreprocessor()
    preprocessor.run()

    feature_engineer = features.FeatureEngineer()
    feature_engineer.run()

    model_trainer = model.ModelTrainer()
    model_trainer.run()

if __name__ == "__main__":
    main()
