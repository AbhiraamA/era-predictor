import pandas as pd
import numpy as np
from pybaseball import pitching_stats
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

START = 2000
END = 2024

pitching = pitching_stats(START, END)
pitching.to_csv("pitching.csv")
pitching = pitching.groupby("IDfg", group_keys=False).filter(lambda x: x.shape[0] > 1)

def next_ssn(plyr):
    plyr = plyr.sort_values("Season")
    plyr["Next_ERA"] = plyr["ERA"].shift(-1)
    return plyr

pitching = pitching.groupby("IDfg", group_keys=False).apply(next_ssn)

cnt_null = pitching.isnull().sum()
full_cols = list(pitching.columns[cnt_null == 0])
pitching = pitching[full_cols + ["Next_ERA"]].copy()

pitching = pitching.drop(columns=["Dollars", "Age Rng"], errors="ignore")
pitching["Team"] = pitching["Team"].astype("category").cat.codes

pitch_2 = pitching.copy()
pitching = pitching.dropna().copy()

col_to_rem = ["Next_ERA", "Name", "Team", "IDfg", "Season"]
sel_col = pitching.columns[~pitching.columns.isin(col_to_rem)]

# Ensure columns are float64 before scaling
pitching[sel_col] = pitching[sel_col].astype('float64')

scaler = MinMaxScaler()
pitching.loc[:, sel_col] = scaler.fit_transform(pitching[sel_col])

rr = Ridge(alpha=1)  # setting higher reduces overfitting

split = TimeSeriesSplit(n_splits=3)  # split data into 3 parts in a time series way

sfs = SequentialFeatureSelector(rr, n_features_to_select=20, direction="forward", cv=split, n_jobs=4)

sfs.fit(pitching[sel_col], pitching["Next_ERA"])
preds = list(sel_col[sfs.get_support()])

def backetest(data, model, pred, start=5, step=1):
    all_preds = []
    yrs = sorted(data["Season"].unique())

    for i in range(start, len(yrs), step):
        curr_year = yrs[i]
        train = data[data["Season"] < curr_year]
        test = data[data["Season"] == curr_year]

        model.fit(train[pred], train["Next_ERA"])

        prediction = model.predict(test[pred])

        prediction = pd.Series(prediction, index=test.index)
        comb = pd.concat([test["Next_ERA"], prediction], axis=1)  # each series should be treated as a separate column
        comb.columns = ["actual", "prediction"]
        all_preds.append(comb)
    return pd.concat(all_preds)

predictions = backetest(pitching, rr, preds)

mse = mean_squared_error(predictions["actual"], predictions["prediction"])

def plyr_hist(df):
    df = df.sort_values("Season")
    df["player_season"] = range(0, df.shape[0])

    # Calculate the expanding correlation
    expanding_corr = df[["player_season", "ERA"]].expanding().corr().loc[(slice(None), "player_season"), "ERA"]

    # Reset index and avoid duplicate labels by using groupby with cumcount
    expanding_corr.index = df.index
    df["ERA_corr"] = expanding_corr
    df["ERA_corr"] = df["ERA_corr"].fillna(1)

    df["ERA_difference"] = df["ERA"] / df["ERA"].shift(1)
    df["ERA_difference"] = df["ERA_difference"].fillna(1)
    df.loc[df["ERA_difference"] == np.inf, "ERA_difference"] = 1

    return df

pitching = pitching.groupby("IDfg", group_keys=False).apply(plyr_hist)

def group_averages(df):
    return df["ERA"] / df["ERA"].mean()

pitching["ERA_season"] = pitching.groupby("Season", group_keys=False).apply(group_averages)

new_pred = preds + ["player_season", "ERA_corr", "ERA_season", "ERA_difference"]
predictions = backetest(pitching, rr, new_pred)
mse = mean_squared_error(predictions["actual"], predictions["prediction"])

merge = predictions.merge(pitching, left_index=True, right_index=True)
merge["diff"] = (predictions["actual"] - predictions["prediction"]).abs()

merge_2023 = merge[merge["Season"] == 2023]

selected_columns = ["IDfg", "Season", "Name", "ERA", "Next_ERA", "diff", "prediction", "actual"]

pred_2024 = merge_2023[selected_columns].sort_values(by=["diff"])
print(pred_2024)
