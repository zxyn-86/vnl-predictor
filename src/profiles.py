import numpy as np
import pandas as pd

def get_team_profile(df, team):
    # for custom matchups getting historical features for each team
    team_data = df[(df["Team"] == team)].sort_values(["Y/M/D", "Match_ID"])
    latest = team_data.iloc[-1]

    return {
        "Prior_Win_Rate": latest["Prior_Win_Rate"],
        "Recent_Form": latest["Recent_Form"],
        "Prior_ATTACK_Efficiency": latest["Prior_ATTACK_Efficiency"],
        "Prior_SERVE_Efficiency": latest["Prior_SERVE_Efficiency"],
        "Prior_RECEPTION_Efficiency": latest["Prior_RECEPTION_Efficiency"],
        "Prior_BLOCK_Efficiency": latest["Prior_BLOCK_Efficiency"],
        "Prior_DIG_Error_Rate": latest["Prior_DIG_Error_Rate"],
        "Prior_SET_Error_Rate": latest["Prior_SET_Error_Rate"],
    }


def get_matchup_features(df, team1, team2):
    team1_profile = get_team_profile(df, team1)
    team2_profile = get_team_profile(df, team2)
    matchup = {}

    for feature in team1_profile.keys():
        matchup[f"{feature}_Diff"] = team1_profile[feature] - team2_profile[feature]

    matchup_df = pd.DataFrame([matchup])

    return matchup_df





    

                   
