from pathlib import Path

import numpy as np
import pandas as pd

#setting up file paths for cleaned and feature engineered data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vnl_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "features" / "vnl_features.csv"

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



def calculate_historical_features(team_data: pd.DataFrame) -> pd.DataFrame:
	team_data = team_data.sort_values(["Y/M/D", "Match_ID"]).copy()
	history = []
	results = []

#creating historical features for each match based on the teams 40 prev matches
#this is individual to each team 
# we initialise some values with Nan as these would be the initial matches where there isnt any match history 
	for _, date_matches in team_data.groupby("Y/M/D", sort=True):
		if history:
			history_df = pd.DataFrame(history)
			historical_values = {
				"Prior_Win_Rate": history_df["Won"].mean(),
				"Recent_Form": history_df["Won"].tail(40).mean(),
			}
			for feature in PERFORMANCE_FEATURES:
				historical_values[f"Prior_{feature}"] = (
					history_df[feature].tail(40).mean()
				)
		else:
			historical_values = {
				"Prior_Win_Rate": np.nan,
				"Recent_Form": np.nan,
			}
			for feature in PERFORMANCE_FEATURES:
				historical_values[f"Prior_{feature}"] = np.nan

		for _, row in date_matches.iterrows():
			row_dict = row.to_dict()
			row_dict.update(historical_values)
			results.append(row_dict)

		for _, row in date_matches.iterrows():
			history.append(row[["Won"] + PERFORMANCE_FEATURES].to_dict())

	return pd.DataFrame(results)

#here we drop all the rows where we couldnt calculate historical features for the teams as these would be the first matches of the season 
#wheras in the above func we just initialise the values with Nan for the first matches of the season
def build_features(input_path: Path = INPUT_PATH) -> pd.DataFrame:
	data = pd.read_csv(input_path)
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
		calculate_historical_features(team_data)
		for _, team_data in data.groupby("Team")
	]
	data = pd.concat(historical, ignore_index=True)
	data = data.sort_values(["Y/M/D", "Match_ID", "Team"]).reset_index(drop=True)

	team_data = data[["Match_ID", "Y/M/D", "Team", "VS_Team", "Won"] + PRIOR_FEATURES]
	opponent_data = data[["Match_ID", "Team"] + PRIOR_FEATURES].rename(
		columns={
			"Team": "VS_Team",
			**{feature: f"VS_{feature}" for feature in PRIOR_FEATURES},
		}
	)
	match_data = team_data.merge(
		opponent_data,
		on=["Match_ID", "VS_Team"],
		how="left",
	)

	for feature in PRIOR_FEATURES:
		match_data[f"{feature}_Diff"] = (
			match_data[feature] - match_data[f"VS_{feature}"]
		)

	return match_data.dropna(subset=FEATURE_COLUMNS)


def main() -> None:
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	match_data = build_features()
	match_data.to_csv(OUTPUT_PATH, index=False)
	print(f"Saved {len(match_data)} feature rows to {OUTPUT_PATH}")
	print(f"Feature columns: {FEATURE_COLUMNS}")


if __name__ == "__main__":
	main()
