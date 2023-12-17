import time
import pandas as pd
from src.config import BASE_DIR

def clean():
    df_train = pd.read_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_training-set.csv')
    df_test = pd.read_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_testing-set.csv')

    del df_train["id"]
    del df_test["id"]

    print(f"{df_train.shape = }")
    print(df_train.columns)
    print(df_train.dtypes)
    print(df_train["proto"].unique())
    print(len(df_train["proto"].unique()))
    print([proto for proto in df_test["proto"].unique() if proto not in df_train["proto"].unique()])
    print(len(df_test["proto"].unique()))
    print(df_train["state"].unique())
    print(len(df_train["state"].unique()))
    print([state for state in df_test["state"].unique() if state not in df_train["state"].unique()])
    print(len(df_test["state"].unique()))
    print(df_train["attack_cat"].unique())
    print(len(df_train["attack_cat"].unique()))
    print([attack_cat for attack_cat in df_test["attack_cat"].unique() if attack_cat not in df_train["attack_cat"].unique()])
    print(len(df_test["attack_cat"].unique()))
    
    # print(df_train["proto"].head())
    # print(df_test["proto"].head())

    
    unique_proto = list(df_train["proto"].unique())
    [unique_proto.append(proto) for proto in df_test["proto"].unique() if proto not in unique_proto]
    df_train["proto"] = df_train["proto"].apply(lambda x: unique_proto.index(x))
    df_test["proto"] = df_test["proto"].apply(lambda x: unique_proto.index(x))

    # print(df_train["proto"].head())
    # print(df_test["proto"].head())
    
    unique_state = list(df_train["state"].unique())
    [unique_state.append(state) for state in df_test["state"].unique() if state not in unique_state]
    df_train["state"] = df_train["state"].apply(lambda x: unique_state.index(x))
    df_test["state"] = df_test["state"].apply(lambda x: unique_state.index(x))

    unique_attack_cat = list(df_train["attack_cat"].unique())
    [unique_attack_cat.append(attack_cat) for attack_cat in df_test["attack_cat"].unique() if attack_cat not in unique_attack_cat]
    df_train["attack_cat"] = df_train["attack_cat"].apply(lambda x: unique_attack_cat.index(x))
    df_test["attack_cat"] = df_test["attack_cat"].apply(lambda x: unique_attack_cat.index(x))

    df_train.to_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_training-set_cleaned.csv', float_format='%f', index=False)
    df_test.to_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_testing-set_cleaned.csv', float_format='%f', index=False)

if __name__=="__main__":
    start = time.time()
    clean()
    print(f"time elapsed: {time.time()-start}s")