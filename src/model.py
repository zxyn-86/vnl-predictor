from pathlib import Path

import pandas as pd
import joblib
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

from features import FeatureEngineer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "features" / "vnl_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ridge_classifier_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


class ModelTrainer:
	"""Trains and evaluates the volleyball match prediction model."""

	def __init__(self, input_path: Path = INPUT_PATH, alpha: float = 1.0):
		self.input_path = input_path
		self.alpha = alpha
		self.feature_columns = FeatureEngineer.FEATURE_COLUMNS
		self.model = None
		self.scaler = None
		self.metrics = None

	def train_and_evaluate(self) -> dict:
		"""Train model and evaluate on test set."""
		data = pd.read_csv(self.input_path)
		data["Y/M/D"] = pd.to_datetime(data["Y/M/D"])

		training_data = data[data["Y/M/D"].dt.year < 2025]
		testing_data = data[data["Y/M/D"].dt.year == 2025]

		x_train = training_data[self.feature_columns]
		y_train = training_data["Won"]
		x_test = testing_data[self.feature_columns]
		y_test = testing_data["Won"]

		self.scaler = StandardScaler()
		x_train_scaled = self.scaler.fit_transform(x_train)
		x_test_scaled = self.scaler.transform(x_test)

		self.model = RidgeClassifier(alpha=self.alpha)
		self.model.fit(x_train_scaled, y_train)
		predictions = self.model.predict(x_test_scaled)

		self.metrics = {
			"accuracy": accuracy_score(y_test, predictions),
			"precision": precision_score(y_test, predictions, zero_division=0),
			"recall": recall_score(y_test, predictions, zero_division=0),
			"f1": f1_score(y_test, predictions, zero_division=0),
		}

		print(f"Training rows: {len(training_data)}")
		print(f"Testing rows: {len(testing_data)}")
		for name, value in self.metrics.items():
			print(f"{name.title()}: {value:.3f}")
		print("\nClassification report:")
		print(classification_report(y_test, predictions, zero_division=0))
		print("Confusion matrix:")
		print(confusion_matrix(y_test, predictions))

		feature_importance = pd.DataFrame(
			{"Feature": self.feature_columns, "Coefficient": self.model.coef_[0]}
		)
		feature_importance["Absolute_Coefficient"] = (
			feature_importance["Coefficient"].abs()
		)

		print("\nFeature importance:")
		print(feature_importance.sort_values("Absolute_Coefficient", ascending=False))

		return {"model": self.model, "scaler": self.scaler, "metrics": self.metrics}

	def save_model(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> None:
		"""Save trained model and scaler to disk."""
		if self.model is None or self.scaler is None:
			raise ValueError("Model and scaler must be trained before saving.")

		model_path.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self.model, model_path)
		joblib.dump(self.scaler, scaler_path)
		print(f"Saved model to {model_path}")
		print(f"Saved scaler to {scaler_path}")

	def run(self) -> dict:
		"""Execute full training pipeline: train, evaluate, and save."""
		result = self.train_and_evaluate()
		self.save_model()
		return result


# Backward compatibility function
def train_and_evaluate(input_path: Path = INPUT_PATH) -> dict:
	trainer = ModelTrainer(input_path=input_path)
	return trainer.train_and_evaluate()


def main() -> None:
	trainer = ModelTrainer()
	trainer.run()


if __name__ == "__main__":
	main()


if __name__ == "__main__":
	train_and_evaluate()
