import pandas as pd
import numpy as np

def preprocess(df, drop_sleep_features=True):
    df = df.copy()

    # 시간 분해
    df["Bedtime"] = pd.to_datetime(df["Bedtime"])
    df["Wakeup time"] = pd.to_datetime(df["Wakeup time"])
    df["Bedtime_hour"] = df["Bedtime"].dt.hour
    df["Wakeup_hour"] = df["Wakeup time"].dt.hour

    # 다중공선성 제거
    df.drop(columns=["Sleep duration", "Deep sleep percentage", "ID", "Bedtime", "Wakeup time"], inplace=True, errors='ignore')

    # 이진 인코딩
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
    df["Smoking status"] = df["Smoking status"].map({"No": 0, "Yes": 1})

    # 이상치 제거 (카페인)
    Q1 = df["Caffeine consumption"].quantile(0.25)
    Q3 = df["Caffeine consumption"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Caffeine consumption"] >= Q1 - 1.5 * IQR) & 
            (df["Caffeine consumption"] <= Q3 + 1.5 * IQR)]

    # 결측값 처리 (성별별 중앙값)
    df["Caffeine consumption"] = df.groupby("Gender")["Caffeine consumption"].transform(lambda x: x.fillna(x.median()))
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    if drop_sleep_features:
        df.drop(columns=["REM sleep percentage", "Light sleep percentage"], inplace=True, errors='ignore')

    return df

