import pandas as pd
import numpy as np
import math
import sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import est_ps_LR2
import est_ate
import nn
def calc_treatment(df_treatment, PS):
    df_treatment = df_treatment.copy()
    # df_treatment: treatment group
    df_treatment.loc[:, "denominator"] = df_treatment[["T", PS]].apply(lambda x : x.iloc[0] / x.iloc[1], axis=1)
    df_treatment.loc[:, "numerator"] = df_treatment[["Y", "T", PS]].apply(lambda x: (x.iloc[0] * x.iloc[1]) / x.iloc[2], axis=1)
    return df_treatment["numerator"].sum() / df_treatment["denominator"].sum()
    
def calc_control(df_control, PS):
    df_control = df_control.copy()
    # df_control: control group
    df_control.loc[:, "denominator"] = df_control[["T", PS]].apply(lambda x : (1 - x.iloc[0]) / (1 - x.iloc[1]), axis=1)
    df_control.loc[:, "numerator"] = df_control[["Y", "T", PS]].apply(lambda x: (x.iloc[0] * (1 - x.iloc[1])) / (1 - x.iloc[2]), axis=1)
    return df_control["numerator"].sum() / df_control["denominator"].sum()


def calc_ate_IPW(df, PS):
    df_treatment = df[df["T"] == 1]
    df_control = df[df["T"] == 0]
    ate = calc_treatment(df_treatment, PS) - calc_control(df_control, PS)
    return ate


#逆確率重み付け法（IPTW）
def calc_treatment(df_treatment, PS):
    df_treatment = df_treatment.copy()
    # df_treatment: treatment group
    df_treatment.loc[:, "denominator"] = df_treatment[["T", PS]].apply(lambda x : x.iloc[0] / x.iloc[1], axis=1)
    df_treatment.loc[:, "numerator"] = df_treatment[["Y", "T", PS]].apply(lambda x: (x.iloc[0] * x.iloc[1]) / x.iloc[2], axis=1)
    return df_treatment["numerator"].sum() / df_treatment["denominator"].sum()
    
def calc_control(df_control, PS):
    df_control = df_control.copy()
    # df_control: control group
    df_control.loc[:, "denominator"] = df_control[["T", PS]].apply(lambda x : (1 - x.iloc[0]) / (1 - x.iloc[1]), axis=1)
    df_control.loc[:, "numerator"] = df_control[["Y", "T", PS]].apply(lambda x: (x.iloc[0] * (1 - x.iloc[1])) / (1 - x.iloc[2]), axis=1)
    return df_control["numerator"].sum() / df_control["denominator"].sum()

def calc_ate_IPW(df, PS):
    df_treatment = df[df["T"] == 1]
    df_control = df[df["T"] == 0]
    ate = calc_treatment(df_treatment, PS) - calc_control(df_control, PS)
    return ate

def calc_ate_IPW2(df, PS):
    df=df.copy()
# Y, T, PS（傾向スコア）の列を使用してATEを計算
    df['ATE_i'] = (df['Y'] / df[PS]) * df['T'] - (df['Y'] / (1 - df[PS])) * (1 - df['T'])
    
    # ATEの平均を計算
    ate = df['ATE_i'].mean()
    return ate, df['ATE_i']
    #return ate

from sklearn.linear_model import LogisticRegression
import pandas as pd
def est_ps_LR(df_est, name):
    x = df_est[["X1", "X2", "X3", "X4"]]
    y = df_est["T"]
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x, y)
    # 傾向スコアの予測
    est_ps = logistic_model.predict_proba(x)[:, 1]
    df_est_ps = pd.DataFrame({name: est_ps})
    return df_est_ps

def est_ps_LR2(df_est,df_test, name, kl):
    x = df_est[["X1", "X2"]]
    x_test = df_test[["X1", "X2"]]
    kl_test = kl[["X1", "X2"]]
    y = df_est["T"]
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x, y)
    # 傾向スコアの予測
    est_ps = logistic_model.predict_proba(x_test)[:, 1]
    kl1 = logistic_model.predict_proba(kl_test)[:, 1]
    
    df_est_ps = pd.DataFrame({name: est_ps})
    kl_result = pd.DataFrame({name: kl1})
    return df_est_ps, kl_result

# def est_ps_LR(df_est, name):
#     df_est = df_est.copy()
#     # (todo) どうやって学習と予測を行う?
#     lr = LogisticRegression(max_iter=1000)
#     x = df_est[["X1", "X2", "X3", "X4"]]
#     y = df_est["T"]
#     lr.fit(x, y)
#     est_ps =  lr.predict_proba(x)[:,1]
#     df_est_ps = pd.DataFrame({name: est_ps})
#     return df_est_ps


        df = pd.read_csv(filename)
        df_test = pd.read_csv(test)
        df_kl = pd.read_csv(kl_nony)
        
        ate = 0.0
        ate_LR = 0.0

        # load data for estimating the propensity score (PS)
        df_est = make_data(df)
        df_est_test = make_data(df_test)
        # estimate average treatment effects using true PS
        ate = calc_ate_IPW(df_est, "truePS")
        ate_list = np.append(ate_list, ate)

        # estimate the propensity score using a logistic regression
        name = "PS_NN"
        
        # df_est_ps_linear, kl = nn.est_ps_NN3(df_est, df_est_test, name,df_kl )
        df_est_ps_linear, kl = est_ps_LR2.est_ps_LR2(df_est, df_est_test, name,df_kl)
        # print(kl)
        print((df_est_ps_linear).max())
        print("max-ps" + str(i) +" : " + str(df_est_ps_linear.max()))
        print("mix-ps" + str(i) +" : " + str(df_est_ps_linear.min()))
        df_add_ps_linear = pd.concat([df_est_test, df_est_ps_linear], axis=1)
        kl = pd.concat([df_kl, kl], axis=1)
            # CSVファイルに追記
        #append_to_csv(df_est_ps_linear, 'LR_ps.csv')
        # print(kl)
        ate_LR_linear = est_ate.calc_ate_IPW(df_add_ps_linear, name)
        ate2_LR_linear, a = est_ate.calc_ate_IPW2(df_add_ps_linear, name)
        # print(ate2_LR_linear)
        ate_list_LR_linear = np.append(ate_list_LR_linear, ate_LR_linear)
        ate2_list_LR_linear = np.append(ate2_list_LR_linear, ate2_LR_linear)
        #append_to_csv(ate2_LR_linear, 'LR_ate.csv')
