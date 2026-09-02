from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "vnl_features.csv"


def train_and_evaluate(input_path: Path = INPUT_PATH) -> dict:
	data = pd.read_csv(input_path)
	data["Y/M/D"] = pd.to_datetime(data["Y/M/D"])

	training_data = data[data["Y/M/D"].dt.year < 2025]
	testing_data = data[data["Y/M/D"].dt.year == 2025]

	x_train = training_data[FEATURE_COLUMNS]
	y_train = training_data["Won"]
	x_test = testing_data[FEATURE_COLUMNS]
	y_test = testing_data["Won"]

	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	model = RidgeClassifier(alpha=5)
	model.fit(x_train_scaled, y_train)
	predictions = model.predict(x_test_scaled)

	metrics = {
		"accuracy": accuracy_score(y_test, predictions),
		"precision": precision_score(y_test, predictions, zero_division=0),
		"recall": recall_score(y_test, predictions, zero_division=0),
		"f1": f1_score(y_test, predictions, zero_division=0),
	}

	print(f"Training rows: {len(training_data)}")
	print(f"Testing rows: {len(testing_data)}")
	for name, value in metrics.items():
		print(f"{name.title()}: {value:.3f}")
	print("\nClassification report:")
	print(classification_report(y_test, predictions, zero_division=0))
	print("Confusion matrix:")
	print(confusion_matrix(y_test, predictions))

	feature_importance = pd.DataFrame(
		{"Feature": FEATURE_COLUMNS, "Coefficient": model.coef_}
	)
	feature_importance["Absolute_Coefficient"] = (
		feature_importance["Coefficient"].abs()
	)

    
    
	print("\nFeature importance:")
	print(feature_importance.sort_values("Absolute_Coefficient", ascending=False))

	return {"model": model, "scaler": scaler, "metrics": metrics}


if __name__ == "__main__":
	train_and_evaluate()
