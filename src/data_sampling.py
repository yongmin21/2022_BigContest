from random import Random
import config

import argparse
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

def run(method):
    
    # 학습 데이터 불러오기
    df = pd.read_parquet(config.TRAINING_FILE)

    # 랜덤언더샘플링 적용하기
    if method == "random":
        rus = RandomUnderSampler(random_state=0)
    else:
        print("RandomUnderSampler Default")
        rus = RandomUnderSampler(random_state=0)

    X, y = df.drop(['is_applied'], axis=1), df['is_applied']
    del df
    X_resampled, y_resampled = rus.fit_resample(X, y)


    print(y.mean())
    print(pd.Series(y_resampled).mean())

if __name__ == "__main__":
    # argparse 초기화
    parser = argparse.ArgumentParser()

    # 필요한 입력 변수와 타입 추가
    parser.add_argument(
        "--method",
        type=str
    )

    # 입력 변수를 읽어오기
    args = parser.parse_args()

    # 콘솔로 읽은 값에 대해 코드 실행
    run(method=args.method)
