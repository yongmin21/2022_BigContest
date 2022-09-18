from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import pandas as pd
import config

def get_nan_table(df, index):
    nan_df = (pd.DataFrame(
        {'total_nan': df.isnull().sum(),
         'ratio': df.isnull().sum() / df.shape[0]},
         index=index
    ))
    return nan_df

def impute_with_logic(df):
    df.fillna({'personal_rehabilitation_yn': 0,
               'personal_rehabilitation_complete_yn' : -1,
               'existing_loan_cnt': 0
               }, inplace=True)

    (df.loc[df['existing_loan_cnt']==0, 'existing_loan_amt']
     .fillna(0, inplace=True))

    df['existing_loan_amt'].fillna(18000000., inplace=True) 
    # 기대출금액의 결측값은 모두 기대출수가 1인 값들로부터 나왔기 때문에, 기대출수가 1인 값들의 기대출금액 중앙값으로 대치

if __name__ == "__main__":
    # Configure
    DIR = config.DATA

    # Load Data

    df = pd.read_parquet(DIR + "data_2022_09_19.parquet")
    meta = pd.read_pickle(DIR + "meta.pkl")

    train = df[df['loanapply_insert_time'] < '2022-06-01']
    test = df[df['loanapply_insert_time'] >= '2022-06-01']

    print("Data Loaded")
    print("===========")
    # 
    print(get_nan_table(train, meta.index))
    impute_with_logic(train)
    impute_with_logic(test)

    # Impute
    print("Start Impute\n")
    int_v = meta[(meta.level == 'interval') & (meta.keep)].index
    bin_v = ['gender'] # 카테고리, 이진 변수에서의 결측값은 gender 하나뿐

    train_impute = train.copy()
    test_impute = test.copy()

    simp_freq = SimpleImputer(strategy = 'most_frequent') # 카테고리형 - 최빈값으로 대치
    simp_med = SimpleImputer(strategy="mean") # 수치형 - 평균값으로 대치

    train_impute[int_v] = simp_med.fit_transform(train_impute[int_v])
    test_impute[int_v] = simp_med.transform(test_impute[int_v])

    train_impute[bin_v] = simp_freq.fit_transform(train_impute[bin_v])
    test_impute[bin_v] = simp_freq.fit_transform(test_impute[bin_v])

    print(get_nan_table(train_impute, meta.index))

    # 칼럼 제거
    train_impute.drop(["loanapply_insert_time", "insert_time", "user_id", "application_id"], axis=1, inplace=True)
    test_impute.drop(["loanapply_insert_time", "insert_time", "user_id", "application_id"], axis=1, inplace=True)

    train_impute.to_parquet(DIR + "train_impute.parquet")
    test_impute.to_parquet(DIR + "test_impute.parquet")

    #iterative = IterativeImputer(random_state=0) # 수치형 - 모델로 예측하여 대치

