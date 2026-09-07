from pathlib import Path

import numpy as np
import pandas as pd

# setting up file paths for cleaned and feature engineered data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vnl_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "vnl_features.csv"


class FeatureEngineer:
	"""Builds historical and opponent-difference features for match prediction."""

	PERFORMANCE_FEATURES = [
		"ATTACK_Efficiency",
		"SERVE_Efficiency",
		"RECEPTION_Efficiency",
		"BLOCK_Efficiency",
		"DIG_Error_Rate",
		"SET_Error_Rate",
	]

	PRIOR_FEATURES = [
		"Prior_Win_Rate",
		"Recent_Form",
		"Prior_ATTACK_Efficiency",
		"Prior_SERVE_Efficiency",
		"Prior_RECEPTION_Efficiency",
		"Prior_BLOCK_Efficiency",
		"Prior_DIG_Error_Rate",
		"Prior_SET_Error_Rate",
	]

	FEATURE_COLUMNS = [f"{feature}_Diff" for feature in PRIOR_FEATURES]

	def __init__(self, input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH):
		self.input_path = input_path
		self.output_path = output_path
		self.window = 40  # Number of previous matches to consider

	def _calculate_historical_features(self, team_data: pd.DataFrame) -> pd.DataFrame:
		"""Calculate historical features for a single team."""
		team_data = team_data.sort_values(["Y/M/D", "Match_ID"]).copy()
		history = []
		results = []

		for _, date_matches in team_data.groupby("Y/M/D", sort=True):
			if history:
				history_df = pd.DataFrame(history)
				historical_values = {
					"Prior_Win_Rate": history_df["Won"].mean(),
					"Recent_Form": history_df["Won"].tail(self.window).mean(),
				}
				for feature in self.PERFORMANCE_FEATURES:
					historical_values[f"Prior_{feature}"] = (
						history_df[feature].tail(self.window).mean()
					)
			else:
				historical_values = {
					"Prior_Win_Rate": np.nan,
					"Recent_Form": np.nan,
				}
				for feature in self.PERFORMANCE_FEATURES:
					historical_values[f"Prior_{feature}"] = np.nan

			for _, row in date_matches.iterrows():
				row_dict = row.to_dict()
				row_dict.update(historical_values)
				results.append(row_dict)

			for _, row in date_matches.iterrows():
				history.append(row[["Won"] + self.PERFORMANCE_FEATURES].to_dict())

		return pd.DataFrame(results)

	def build_features(self) -> pd.DataFrame:
		"""Build all features from cleaned data."""
		data = pd.read_csv(self.input_path)
		data["Y/M/D"] = pd.to_datetime(data["Y/M/D"])
		data["DIG_Error_Rate"] = np.where(
			data["DIG_Digs"] > 0,
			data["DIG_Errors"] / data["DIG_Digs"],
			np.nan,
		)
		data["SET_Error_Rate"] = np.where(
			data["SET_Attempts"] > 0,
			data["SET_Errors"] / data["SET_Attempts"],
			np.nan,
		)

		historical = [
			self._calculate_historical_features(team_data)
			for _, team_data in data.groupby("Team")
		]
		data = pd.concat(historical, ignore_index=True)
		data = data.sort_values(["Y/M/D", "Match_ID", "Team"]).reset_index(drop=True)

		team_data = data[["Match_ID", "Y/M/D", "Team", "VS_Team", "Won"] + self.PRIOR_FEATURES]
		opponent_data = data[["Match_ID", "Team"] + self.PRIOR_FEATURES].rename(
			columns={
				"Team": "VS_Team",
				**{feature: f"VS_{feature}" for feature in self.PRIOR_FEATURES},
			}
		)
		match_data = team_data.merge(
			opponent_data,
			on=["Match_ID", "VS_Team"],
			how="left",
		)

		for feature in self.PRIOR_FEATURES:
			match_data[f"{feature}_Diff"] = (
				match_data[feature] - match_data[f"VS_{feature}"]
			)

		return match_data.dropna(subset=self.FEATURE_COLUMNS)

	def save_features(self, data: pd.DataFrame) -> None:
		"""Save feature data to CSV."""
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		data.to_csv(self.output_path, index=False)
		print(f"Saved {len(data)} feature rows to {self.output_path}")
		print(f"Feature columns: {self.FEATURE_COLUMNS}")

	def run(self) -> pd.DataFrame:
		"""Execute the full feature engineering pipeline: build and save."""
		data = self.build_features()
		self.save_features(data)
		return data


# Module-level constants for backward compatibility
PERFORMANCE_FEATURES = FeatureEngineer.PERFORMANCE_FEATURES
PRIOR_FEATURES = FeatureEngineer.PRIOR_FEATURES
FEATURE_COLUMNS = FeatureEngineer.FEATURE_COLUMNS


def main() -> None:
	engineer = FeatureEngineer()
	engineer.run()


if __name__ == "__main__":
	main()
