import config
import joblib

import argparse
import os

import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#from sklearn.ensemble import RandomForestClassifier


def run(fold):
    # fold data load
    df = pd.read_parquet(config.TRAINING_FILE)

    # train kfold != fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # valid kfold == fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # train, valid split
    x_train = df_train.drop("is_applied", axis=1).values
    y_train = df_train.is_applied.values

    x_valid = df_valid.drop("is_applied", axis=1).values
    y_valid = df_valid.is_applied.values

    # model init
    #rf = RandomForestClassifier(n_jobs=-1, random_state=0)
    model = xgb.XGBClassifier(random_state=0, tree_method='gpu_hist', gpu_id=0)

    print("==== fit model ====")
    model.fit(x_train, y_train)

    print("==== predict model ====")
    y_preds = model.predict(x_valid)

    # calculate score
    print("==== calculate f1_score ====")
    score = f1_score(y_valid, y_preds)
    print(f"Fold={fold}, F1_Score={score}")

    # model save
    joblib.dump(
        model,
        os.path.join(config.MODEL_OUTPUT, f"xgb_{fold}.bin")
    )

if __name__ == "__main__":
    # init argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    # run model
    run(
        fold=args.fold
        #model=args.model
    )



