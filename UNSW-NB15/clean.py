import time, pandas as pd, numpy as np 
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss 
from src.config import BASE_DIR

def clean():
    df_train = pd.read_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_training-set.csv', na_values="-")
    df_test = pd.read_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_testing-set.csv', na_values="-")

    del df_train["id"]
    del df_test["id"]

    print(df_train.columns)

    print(len(df_train.columns.difference(["service", "proto", "state", "attack_cat", "label"])))
    

    for column in df_train.columns:
        print(f"{column}: {df_train[column].isnull().sum()}")
        print(f"{column}: {df_test[column].isnull().sum()}")
    
    for column in ("proto", "state", "service"):
        print(df_train[column].value_counts())

    numeric_cols = df_train.columns.difference(["service", "proto", "state", "attack_cat", "label"])
    scaler = MinMaxScaler()
    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")
    df_train_trans_cats = enc.fit_transform(df_train[["service", "proto", "state"]])
    df_test_trans_cats = enc.transform(df_test[["service",  "proto", "state"]])

    df_train = pd.concat([df_train, df_train_trans_cats], axis=1).drop(["service", "proto", "state"], axis=1)
    df_test = pd.concat([df_test, df_test_trans_cats], axis=1).drop(["service", "proto", "state"], axis=1)

    print(df_train["attack_cat"].value_counts())

    enc = OrdinalEncoder(min_frequency=0.05).set_output(transform="pandas")
    df_train["attack_cat"] = enc.fit_transform(df_train[["attack_cat"]])
    df_test["attack_cat"] = enc.transform(df_test[["attack_cat"]])

    df_train["attack_cat"] = df_train["attack_cat"].astype(int)
    df_test["attack_cat"] = df_test["attack_cat"].astype(int)

    df_train = df_train[df_train["attack_cat"] != 6]
    df_test = df_test[df_test["attack_cat"] != 6]

    print(df_train["attack_cat"].value_counts())
    print(df_test["attack_cat"].value_counts())
    print(df_train.shape)
    print(df_test.shape)

    df_train.drop_duplicates(inplace=True)
    df_test.drop_duplicates(inplace=True)

    print(df_train["attack_cat"].value_counts())
    print(df_test["attack_cat"].value_counts())
    print(df_train.shape)
    print(df_test.shape)

    df_train_add = df_train[df_train["label"] == 0]
    df_train = pd.concat([df_train, df_train_add])

    print(df_train["attack_cat"].value_counts())
    print(df_test["attack_cat"].value_counts())
    print(df_train.shape)
    print(df_test.shape)

   #  print(df_train["label"].value_counts())

#    #  X = df_train.columns.difference(["label"])
#    #  y = "label"
    
#    #  sampler = SMOTE(sampling_strategy="minority", random_state=13)
#    #  df_train[X], df_train[y] = sampler.fit_resample(df_train[X], df_train[y])
    
#    #  df_train.dropna(inplace=True)

    
#    #  sampler = NearMiss(sampling_strategy='majority') 
#    #  df_train[X], df_train[y] = sampler.fit_resample(df_train[X], df_train[y])
    

#    #  print(df_train["attack_cat"].value_counts())
#    #  print(df_train["label"].value_counts())

#     # # print(df_train["label"].value_counts())

#    #  df_train.dropna(inplace=True)

#    #  sampler = SMOTE(sampling_strategy='minority', random_state=13)
#    #  df_train[X], df_train[y] = sampler.fit_resample(df_train[X], df_train[y])
    
#    #  df_train.dropna(inplace=True)

#    #  print(df_train["attack_cat"].value_counts())


    df_train = df_train[[c for c in df_train if c not in ['attack_cat', 'label']] 
       + ['attack_cat', 'label']]
    df_test = df_test[[c for c in df_test if c not in ['attack_cat', 'label']] 
       + ['attack_cat', 'label']]
    
    
    df_train.to_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_training-set_cleaned.csv', float_format='%f', index=False)
    df_test.to_csv(BASE_DIR / 'UNSW-NB15/data/UNSW_NB15_testing-set_cleaned.csv', float_format='%f', index=False)

if __name__=="__main__":
    start = time.time()
    clean()
    print(f"time elapsed: {time.time()-start}s")