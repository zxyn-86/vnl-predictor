from pathlib import Path

import numpy as np
import pandas as pd

#setting up file paths for raw and processed data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "VNL"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "vnl_cleaned.csv"


# this takes the dataset from github and drops any unnecessary columns and also fills in any blanks ready for feature engineering
def clean_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
	"""Load all VNL match files and create the cleaned match-level dataset."""
	files = sorted(raw_dir.rglob("*.csv"))
	if not files:
		raise FileNotFoundError(f"No CSV files found under {raw_dir}")

	data = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
	data = data[data["SET"] == "ALL"].copy()

	columns_to_drop = ["Match", "Pool", "Week", "City", "Country", "SET"]
	data = data.drop(columns=columns_to_drop, errors="ignore")
	data["Season"] = data["Season"].replace("Men", "2025")
	data["Y/M/D"] = pd.to_datetime(data["Y/M/D"], format="%Y/%m/%d")

#filling blanks as not all matches go to 4 or 5 sets 
	point_columns = [
		"Point1", "Point2", "Point3", "Point4", "Point5",
		"VS_Point1", "VS_Point2", "VS_Point3", "VS_Point4", "VS_Point5",
	]
	data[point_columns] = (
		data[point_columns]
		.replace("-", "0")
		.fillna(0)
		.apply(pd.to_numeric)
	)

	data["Match_ID"] = (
		data["Y/M/D"].astype(str)
		+ "_"
		+ data.apply(
			lambda row: "_".join(sorted([row["Team"], row["VS_Team"]])),
			axis=1,
		)
	)
	data["Won"] = (data["SET_Won"] > data["SET_Lost"]).astype(int)

	data["ATTACK_Efficiency"] = np.where(
		data["ATTACK_Attempts"] > 0,
		(data["ATTACK_Point"] - data["ATTACK_Errors"])
		/ data["ATTACK_Attempts"],
		np.nan,
	)
	data["SERVE_Efficiency"] = np.where(
		data["SERVE_Attempts"] > 0,
		(data["SERVE_Point"] - data["SERVE_Errors"])
		/ data["SERVE_Attempts"],
		np.nan,
	)
	data["RECEPTION_Efficiency"] = np.where(
		data["RECEPTION_Attempts"] > 0,
		(data["RECEPTION_Successful"] - data["RECEPTION_Errors"])
		/ data["RECEPTION_Attempts"],
		np.nan,
	)
	data["DIG_Efficiency"] = np.where(
		data["DIG_Total"] > 0,
		(data["DIG_Digs"] - data["DIG_Errors"]) / data["DIG_Total"],
		np.nan,
	)
	data["SET_Efficiency"] = np.where(
		data["SET_Attempts"] > 0,
		(data["SET_Point"] - data["SET_Errors"]) / data["SET_Attempts"],
		np.nan,
	)
	data["BLOCK_Efficiency"] = np.where(
		data["BLOCK_Touches"] > 0,
		(data["BLOCK_Point"] - data["BLOCK_Errors"]) / data["BLOCK_Touches"],
		np.nan,
	)

	return data.sort_values(["Y/M/D", "Match_ID"]).reset_index(drop=True)


def main() -> None:
	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	data = clean_data()
	data.to_csv(OUTPUT_PATH, index=False)
	print(f"Saved {len(data)} cleaned rows to {OUTPUT_PATH}")


if __name__ == "__main__":
	main()
