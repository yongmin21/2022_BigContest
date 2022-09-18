from ast import Str
import config

import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # train data load
    #df = pd.read_parquet(config.DATA + "train_impute_undersampled.parquet")
    df = pd.read_parquet(config.DATA + "train_impute.parquet")

    # kfold 생성
    df['kfold'] = -1

    # mix data
    df = df.sample(frac=1).reset_index(drop=True)

    # get target variable
    y = df.is_applied.values

    # kfold init
    skf = StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # data save
    df.to_parquet(config.DATA + "train_folds.parquet")