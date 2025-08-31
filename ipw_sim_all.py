import pandas as pd
import numpy as np
import math
import sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys
import est_ps_LR2
import est_ate
import nn


def make_data(df):
    #df_est = df[[ "X1", "X2", "X3", "X4", "Z", "Y", "truePS"]]
    #markov
    df_est = df[[ "X1", "Y", "X2", "Z", "truePS"]]
    #df_est.columns = ["X1", "X2", "X3", "X4", "T", "Y", "truePS"]
    df_est.columns = [ "X1", "Y", "X2", "T", "truePS"]
    return df_est


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
    df_treatment = df[df["T"] == 1.0]#BNのときは1
    df_control = df[df["T"] == 0.0]
    ate = calc_treatment(df_treatment, PS) - calc_control(df_control, PS)
    return ate

def est_rmse(est):
    n = est.shape[0]
    return math.sqrt(np.sum(est ** 2) / n)

def est_rmse_ps(df, est, true):
    df = df.copy()
    n = df.shape[0]
    df.loc[:,"SE-PS"] = df[[est, true]].apply(lambda x : (x.iloc[0] - x.iloc[1]) * (x.iloc[0] - x.iloc[1]), axis=1)
    return math.sqrt(df["SE-PS"].sum() / n)

def est_mae_ps(df, est, true):
    df = df.copy()
    n = df.shape[0]
    df.loc[:,"AE-PS"] = df[[est, true]].apply(lambda x : abs(x.iloc[0] - x.iloc[1]), axis=1)
    return df["AE-PS"].sum() / n

# KLダイバージェンスを計算する関数
def calculate_kl_divergence(kl):
    NV=3
    kl_divergence_0 = 0.0
    kl_divergence_1 = 0.0

    # Tが0のときのKLダイバージェンスを計算
    df_t0 = kl[kl['T'] == 0]
    for i in range(int(np.power(3, NV - 1))):
    # for i in range(int(np.power(2, NV - 1))): #3値のときは(np.power(3, NV - 1))
        p_true_0 = df_t0.iloc[i]['P(T|X1,X2,Y)']
        p_est_0 = 1.0 - df_t0.iloc[i]['PS_NN']
        if p_true_0 > 0 and p_est_0 > 0:  # 0対数エラーを避けるため、0の確率は無視
            kl_divergence_0 += p_est_0 * (np.log(p_est_0) - np.log(p_true_0))
        #print(i,p_est_0)
    #print(df_t0)
    # Tが1のときのKLダイバージェンスを計算
    df_t1 = kl[kl['T'] == 1]
    for i in range(int(np.power(3, NV - 1))):
        p_true_1 = df_t1.iloc[i]['P(T|X1,X2,Y)']
        p_est_1 = df_t1.iloc[i]['PS_NN']
        if p_true_1 > 0 and p_est_1 > 0:  # 0対数エラーを避けるため、0の確率は無視
            kl_divergence_1 += p_est_1 * (np.log(p_est_1) - np.log(p_true_1))

    kl = kl_divergence_0 + kl_divergence_1
    return kl/9.0

    # print(kl["T"])
    # for i in range(int(np.power(3, NV - 1))):
    #     if (kl["T"] == 0.0):
    #         kl["PS_NN"] = 1.0 - kl["PS_NN"]
    #     kl += kl["PS_NN"] * (np.log(kl["PS_NN"]) - np.log(kl["P(T|X1,X2,Y)"]))
    # p_true_0 = prob[i] / prob_marginal[i]  # i番目のパターン時の分類確率 目的変数は0
    # p_est_0 = prob_learn[i] / prob_learn_marginal[i]  # 学習した構造のi番目のパターン時の分類確率 目的変数は0
    # p_true_1 = prob[i + int(np.power(3, NV - 1))] / prob_marginal[i]  # i番目のパターン時の分類確率 目的変数は1
    # p_est_1 = prob_learn[i + int(np.power(3, NV - 1))] / prob_learn_marginal[i]  # 学習した構造のi番目のパターン時の分類確率 目的変数は1
    # print(f"p_true_0 - est_0: {p_true_0:.10f} {p_est_0:.10f}")
    # print(f"p_true_1 - est_1: {p_true_1:.10f} {p_est_1:.10f}")
    # KL += p_est_0 * (np.log(p_est_0) - np.log(p_true_0))
    # KL += p_est_1 * (np.log(p_est_1) - np.log(p_true_1))

    # print(f"KL Divergence: {kl}")

    # # for p_true, p_est in zip(true_probs, estimated_probs):
    # #     if p_true > 0 and p_est > 0:  # 0対数エラーを避けるため、0の確率は無視
    # #         kl_divergence += p_est * (np.log(p_est) - np.log(p_true))
    # return kl
def append_to_csv(df, filename):
    df.to_csv(filename, mode='a', index=False, header=not pd.io.common.file_exists(filename))

def est_ps_rmse2(est, name):
    # Yカラムが1のデータにフィルタリング
    est = est[est["Y"] == 1]

    inner_list = np.empty([0])
    outer_list = np.empty([0])
    # truePSと比較し、大きい・小さい場合の個数とRMSEを計算
    true_ps = est["truePS"]
    est_ps = est[name]
    # truePSが0.5より大きいときの内側寄りリスト（truePSより小さい）と外側寄りリスト（truePSより大きい）
    inner = est_ps[(true_ps > 0.5) & (est_ps < true_ps)] - true_ps[(true_ps > 0.5) & (est_ps < true_ps)]
    outer = est_ps[(true_ps > 0.5) & (est_ps > true_ps)] - true_ps[(true_ps > 0.5) & (est_ps > true_ps)]
    inner_list = np.append(inner_list, inner)
    outer_list = np.append(outer_list, outer)

    # truePSが0.5のときの大きいリストと小さいリスト
    equal_to_half_greater = est_ps[true_ps == 0.5][est_ps > true_ps] - true_ps[true_ps == 0.5][est_ps > true_ps]
    equal_to_half_smaller = est_ps[true_ps == 0.5][est_ps < true_ps] - true_ps[true_ps == 0.5][est_ps < true_ps] 

    # truePSが0.5より小さいときの内側寄りリスト（truePSより大きい）と外側寄りリスト（truePSより小さい）
    inner = est_ps[(true_ps < 0.5) & (est_ps > true_ps)] - true_ps[(true_ps < 0.5) & (est_ps > true_ps)]
    outer = est_ps[(true_ps < 0.5) & (est_ps < true_ps)] - true_ps[(true_ps < 0.5) & (est_ps < true_ps)]
    inner_list = np.append(inner_list, inner)
    outer_list = np.append(outer_list, outer)

    # # 大きい場合と小さい場合のリスト作成
    # greater_diff_LR = est_ps[est_ps > true_ps]
    # smaller_diff_LR = est_ps[est_ps < true_ps]

    inner_count = len(inner_list) 
    outer_count = len(outer_list) 
    equal_half_greater_count = len(equal_to_half_greater)
    equal_half_smaller_count = len(equal_to_half_smaller)
    # # 大きい場合と小さい場合の個数を計算
    # greater_count = len(greater_diff_LR)
    # smaller_count = len(smaller_diff_LR)
    
    # RMSEを計算（リストが空の場合は0に設定）
    if inner_count > 0:
        inner_rmse = np.sqrt(((inner_list)**2).mean())
    else:
        inner_rmse = 0

    if outer_count > 0:
        outer_rmse = np.sqrt(((outer_list)**2).mean())
    else:
        outer_rmse = 0

    if equal_half_greater_count > 0:
        equal_half_greater_rmse = np.sqrt(((equal_to_half_greater)**2).mean())
    else:
        equal_half_greater_rmse = 0

    if equal_half_smaller_count > 0:
        equal_half_smaller_rmse = np.sqrt(((equal_to_half_smaller)**2).mean())
    else:
        equal_half_smaller_rmse = 0



    # 大きい場合と小さい場合のRMSEを計算
    # if greater_count > 0:
    #     greater_rmse = np.sqrt(((greater_diff_LR - true_ps[est_ps > true_ps])**2).mean())
    # else:
    #     greater_rmse = 0
    # if smaller_count > 0:
    #     smaller_rmse = np.sqrt(((smaller_diff_LR - true_ps[est_ps < true_ps])**2).mean())
    # else:
    #     smaller_rmse = 0
    # print(f"Greater count: {greater_count}, Greater RMSE: {greater_rmse}")
    # print(f"Smaller count: {smaller_count}, Smaller RMSE: {smaller_rmse}")
    # print(f"Inner count: {inner_count}, Inner RMSE: {inner_rmse}")
    # print(f"Outer count: {outer_count}, Outer RMSE: {outer_rmse}")
    # print(f"Equal to 0.5, Greater count: {equal_half_greater_count}, Greater RMSE: {equal_half_greater_rmse}")
    # print(f"Equal to 0.5, Smaller count: {equal_half_smaller_count}, Smaller RMSE: {equal_half_smaller_rmse}")
    return inner_count, inner_rmse, outer_count, outer_rmse, equal_half_greater_count, equal_half_greater_rmse, equal_half_smaller_count, equal_half_smaller_rmse

def main(args):
    ate_list = np.empty([0])
    ate_list_LR_linear = np.empty([0])
    ate2_list_LR_linear = np.empty([0])
    rmse_ps_LR_linear = np.empty([0])
    mae_ps_LR_linear = np.empty([0])
    kldiv1_list = np.empty([0])
    kldiv0_list = np.empty([0])
    kldiv_list = np.empty([0])
    ate_list_LR_nonlinear = np.empty([0])
    ate2_list_LR_nonlinear = np.empty([0])
    rmse_ps_LR_nonlinear = np.empty([0])
    mae_ps_LR_nonlinear = np.empty([0])
    ate_list_GBN_linear = np.empty([0])
    ate2_list_GBN_linear = np.empty([0])
    rmse_ps_GBN_linear = np.empty([0])
    mae_ps_GBN_linear = np.empty([0])
    ate_list_GBN_nonlinear = np.empty([0])
    ate2_list_GBN_nonlinear = np.empty([0])
    rmse_ps_GBN_nonlinear = np.empty([0])
    mae_ps_GBN_nonlinear = np.empty([0])
    ate_list_NCPMIN_linear = np.empty([0])
    ate2_list_NCPMIN_linear = np.empty([0])
    rmse_ps_NCPMIN_linear = np.empty([0])
    mae_ps_NCPMIN_linear = np.empty([0])
    ate_list_NCPMIN_nonlinear = np.empty([0])
    ate2_list_NCPMIN_nonlinear = np.empty([0])
    rmse_ps_NCPMIN_nonlinear = np.empty([0])
    mae_ps_NCPMIN_nonlinear = np.empty([0])
    inner_count_LR_list = np.empty([0]) 
    inner_rmse_LR_list = np.empty([0]) 
    outer_count_LR_list = np.empty([0]) 
    outer_rmse_LR_list = np.empty([0])
    equal_half_greater_count_LR_list = np.empty([0]) 
    equal_half_greater_rmse_LR_list = np.empty([0]) 
    equal_half_smaller_count_LR_list = np.empty([0])
    equal_half_smaller_rmse_LR_list = np.empty([0])
    kl_div=0
    # linear_flag = 0  # 1: linear setting, 0: nonlinear setting
    for i in range(10):
        # データの読み込み
        # filename = "./data/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + ".csv"
        #LRから発生させたデータfilename = "./data/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + "_discret.csv"
        #BNから発生させたデータ
        #filename = "./data/simulation_BN/" + args[1] + "/all/sim_" + args[1] + "_" + str(i + 1) + "_clean.csv"
        #マルコフネットワークから発生させたデータ
        filename = "./markov/data/" + args[1] + "/markov_" + args[1] + "_" + str(i + 1) + ".csv"
        test = "./markov/data/" + args[1] + "/test/markov_" + args[1] + "_" + str(i + 1) + ".csv"
        kl_nony = "./kl_nony.csv"
        # filename = "./markov/data4/" + args[1] + "/markov_" + args[1] + "_" + str(i + 1) + "_4.csv"
        # test = "./markov/data4/" + args[1] + "/test/markov_" + args[1] + "_" + str(i + 1) + "_4.csv"
        # kl_nony = "./markov/kl_nony_ver4_4.csv"
        # GBN_ps_name_linear = "./ps/getBN-probability/result/GBN/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + "_discret_linear_ps.csv"
        # NCPMIN_ps_name_linear = "./ps/getBN-probability/result/NCPMIN/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + "_discret_linear_ps.csv"
        # GBN_ps_name_nonlinear = "./ps/getBN-probability/result/GBN/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + "_discret_nonlinear_ps.csv"
        # NCPMIN_ps_name_nonlinear = "./ps/getBN-probability/result/NCPMIN/test_type2_" + args[1] + "/test_type2_" + args[1] + "_" + str(i + 1) + "_discret_nonlinear_ps.csv"
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
        rmse_ps_linear = est_rmse_ps(df_add_ps_linear, name, "truePS")
        rmse_ps_LR_linear = np.append(rmse_ps_LR_linear, rmse_ps_linear)
        mae_ps_linear = est_mae_ps(df_add_ps_linear, name, "truePS")
        mae_ps_LR_linear = np.append(mae_ps_LR_linear, mae_ps_linear)
        # KL発散の計算
        kldiv = calculate_kl_divergence(kl)
        kldiv_list = np.append(kldiv_list, kldiv)
        print("NNrmse" + str(i+1))
        #print(df_add_ps)     X1    Y   X2    T    truePS     PS_LR
        #print(df_est_ps)       PS_LR
        # sys.exit(1)
        inner_count_LR, inner_rmse_LR, outer_count_LR, outer_rmse_LR, equal_half_greater_count_LR, equal_half_greater_rmse_LR, equal_half_smaller_count_LR, equal_half_smaller_rmse_LR = est_ps_rmse2(df_add_ps_linear,name)        
        inner_count_LR_list = np.append(inner_count_LR_list, inner_count_LR) 
        inner_rmse_LR_list = np.append(inner_rmse_LR_list, inner_rmse_LR) 
        outer_count_LR_list = np.append(outer_count_LR_list, outer_count_LR) 
        outer_rmse_LR_list = np.append(outer_rmse_LR_list, outer_rmse_LR)
        equal_half_greater_count_LR_list = np.append(equal_half_greater_count_LR_list,equal_half_greater_count_LR) 
        equal_half_greater_rmse_LR_list = np.append(equal_half_greater_rmse_LR_list,equal_half_greater_rmse_LR) 
        equal_half_smaller_count_LR_list = np.append(equal_half_smaller_count_LR_list,equal_half_smaller_count_LR)
        equal_half_smaller_rmse_LR_list = np.append(equal_half_smaller_rmse_LR_list,equal_half_smaller_rmse_LR)

        
        # print(df_add_ps_linear)
        kl2 = pd.concat([df_add_ps_linear, a], axis=1)
        # kl2 = pd.merge(a, df_kl, on=['X1', 'X2', 'T'], how='left')
        # print("LR")
        # print(kl2)
        right_2_cols = kl2.iloc[:, -3:-1]
        right_3_cols = kl2.iloc[:, -3:]
        plt.figure()
        # 2段の図を作成 
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        # 上段: 右の2列をプロット（左の列の値を縦軸、右の列の値を横軸に設定） 
        axs[0].scatter(right_2_cols.iloc[:, 1], right_2_cols.iloc[:, 0], s=10) 
        axs[0].set_xlabel(right_2_cols.columns[1]) 
        axs[0].set_ylabel(right_2_cols.columns[0]) 
        axs[0].legend([f"{right_2_cols.columns[0]} vs {right_2_cols.columns[1]}"]) 
        axs[0].set_xlim(0, 1) # 横軸の範囲を0から1に設定
        axs[0].set_ylim(0, 1) 
        # y=x のラインを追加 
        min_val_0 = min(right_2_cols.iloc[:, 1].min(), right_2_cols.iloc[:, 0].min()) 
        max_val_0 = max(right_2_cols.iloc[:, 1].max(), right_2_cols.iloc[:, 0].max()) 
        axs[0].plot([min_val_0, max_val_0], [min_val_0, max_val_0], color='red', label='y=x') 
        axs[0].legend()
        
        # 下段: kl2の一番右の列をプロット 
        axs[1].scatter(right_2_cols.iloc[:, 1], kl2.iloc[:, -1], s=10) 
        axs[1].set_xlabel(right_2_cols.columns[1]) 
        axs[1].set_ylabel(kl2.columns[-1]) 
        axs[1].legend([kl2.columns[-1]], title='b')
        axs[1].set_xlim(0, 1) # 横軸の範囲を0から1に設定
        # y=x のラインを追加 
        min_val_1 = min(right_2_cols.iloc[:, 1].min(), kl2.iloc[:, -1].min()) 
        max_val_1 = max(right_2_cols.iloc[:, 1].max(), kl2.iloc[:, -1].max()) 
        axs[1].legend()
        # plt.savefig('./markov/data4/plot/' + args[1] + '/plotdata4_4_NN_'+ args[1] + '_' +str(i)+'.png', dpi=300, bbox_inches='tight') 
        plt.show()
        plt.close()

        # kldiv1_list = np.append(kldiv1_list, kldiv1)
        # kldiv0_list = np.append(kldiv0_list, kldiv0)
        #print(f"KL Divergence between true PS and estimated PS: {kldiv0, kldiv1}")

        # # estimate the propensity score using a General Bayesian Network
        # name = "PS_GBN"
        # df_est_ps_GBN_linear = pd.read_csv(GBN_ps_name_linear, header=None)
        # df_est_ps_GBN_linear.columns = [name]
        # df_add_ps_GBN_linear = pd.concat([df_est_linear, df_est_ps_GBN_linear], axis=1)
        # ate_GBN_linear = calc_ate_IPW(df_add_ps_GBN_linear, name)
        # ate2_GBN_linear = est_ate.calc_ate_IPW2(df_add_ps_GBN_linear, name)
        # ate_list_GBN_linear = np.append(ate_list_GBN_linear, ate_GBN_linear)
        # ate2_list_GBN_linear = np.append(ate2_list_GBN_linear, ate2_GBN_linear)
        # rmse_ps_linear = est_rmse_ps(df_add_ps_GBN_linear, name, "truePS")
        # rmse_ps_GBN_linear = np.append(rmse_ps_GBN_linear, rmse_ps_linear)
        # mae_ps_linear = est_mae_ps(df_add_ps_GBN_linear, name, "truePS")
        # mae_ps_GBN_linear = np.append(mae_ps_GBN_linear, mae_ps_linear)


        # df_est_ps_GBN_nonlinear = pd.read_csv(GBN_ps_name_nonlinear, header=None)
        # df_est_ps_GBN_nonlinear.columns = [name]
        # df_add_ps_GBN_nonlinear = pd.concat([df_est_nonlinear, df_est_ps_GBN_nonlinear], axis=1)
        # ate_GBN_nonlinear = calc_ate_IPW(df_add_ps_GBN_nonlinear, name)
        # ate2_GBN_nonlinear = est_ate.calc_ate_IPW2(df_add_ps_GBN_nonlinear, name)
        # ate_list_GBN_nonlinear = np.append(ate_list_GBN_nonlinear, ate_GBN_nonlinear)
        # ate2_list_GBN_nonlinear = np.append(ate2_list_GBN_nonlinear, ate2_GBN_nonlinear)
        # rmse_ps_nonlinear = est_rmse_ps(df_add_ps_GBN_nonlinear, name, "truePS")
        # rmse_ps_GBN_nonlinear = np.append(rmse_ps_GBN_nonlinear, rmse_ps_nonlinear)
        # mae_ps_nonlinear = est_mae_ps(df_add_ps_GBN_nonlinear, name, "truePS")
        # mae_ps_GBN_nonlinear = np.append(mae_ps_GBN_nonlinear, mae_ps_nonlinear)

        # # estimate the propensity score using a BNC(AAAI)
        # name = "PS_NCPMIN"
        # df_est_ps_NCPMIN_linear = pd.read_csv(NCPMIN_ps_name_linear, header=None)
        # df_est_ps_NCPMIN_linear.columns = [name]
        # df_add_ps_NCPMIN_linear = pd.concat([df_est_linear, df_est_ps_NCPMIN_linear], axis=1)
        # ate_NCPMIN_linear = calc_ate_IPW(df_add_ps_NCPMIN_linear, name)
        # ate2_NCPMIN_linear = est_ate.calc_ate_IPW2(df_add_ps_NCPMIN_linear, name)
        # ate_list_NCPMIN_linear = np.append(ate_list_NCPMIN_linear, ate_NCPMIN_linear)
        # ate2_list_NCPMIN_linear = np.append(ate2_list_NCPMIN_linear, ate2_NCPMIN_linear)
        # rmse_ps_linear = est_rmse_ps(df_add_ps_NCPMIN_linear, name, "truePS")
        # rmse_ps_NCPMIN_linear = np.append(rmse_ps_NCPMIN_linear, rmse_ps_linear)
        # mae_ps_linear = est_mae_ps(df_add_ps_NCPMIN_linear, name, "truePS")
        # mae_ps_NCPMIN_linear = np.append(mae_ps_NCPMIN_linear, mae_ps_linear)


        # df_est_ps_NCPMIN_nonlinear = pd.read_csv(NCPMIN_ps_name_nonlinear, header=None)
        # df_est_ps_NCPMIN_nonlinear.columns = [name]
        # df_add_ps_NCPMIN_nonlinear = pd.concat([df_est_nonlinear, df_est_ps_NCPMIN_nonlinear], axis=1)
        # ate_NCPMIN_nonlinear = calc_ate_IPW(df_add_ps_NCPMIN_nonlinear, name)
        # ate2_NCPMIN_nonlinear = est_ate.calc_ate_IPW2(df_add_ps_NCPMIN_nonlinear, name)
        # ate_list_NCPMIN_nonlinear = np.append(ate_list_NCPMIN_nonlinear, ate_NCPMIN_nonlinear)
        # ate2_list_NCPMIN_nonlinear = np.append(ate2_list_NCPMIN_nonlinear, ate2_NCPMIN_nonlinear)
        # rmse_ps_nonlinear = est_rmse_ps(df_add_ps_NCPMIN_nonlinear, name, "truePS")
        # rmse_ps_NCPMIN_nonlinear = np.append(rmse_ps_NCPMIN_nonlinear, rmse_ps_nonlinear)
        # mae_ps_nonlinear = est_mae_ps(df_add_ps_NCPMIN_nonlinear, name, "truePS")
        # mae_ps_NCPMIN_nonlinear = np.append(mae_ps_NCPMIN_nonlinear, mae_ps_nonlinear)

    np.set_printoptions(threshold=np.inf)
    print(ate2_list_LR_linear)
    print("truePS")
    # # print("ate_min: " + str(ate_list.min()))
    # # print("ate_max: " + str(ate_list.max()))
    # # print("E[ate]: " + str(ate_list.mean()))
    # # print("ate_std: " + str(ate_list.std()))
    print("Bias: " + str(ate_list.mean() - 0))
    print("RMSE: " + str(est_rmse(ate_list)))
    print("MAE: " + str(np.abs(ate_list).mean()))

    print("linear")
    print("LR-NN")
    # print("ate_min: " + str(ate_list_LR_linear.min()))
    # print("ate_max: " + str(ate_list_LR_linear.max()))
    # print("E[ate]: " + str(ate_list_LR_linear.mean()))
    # print("ate_std: " + str(ate_list_LR_linear.std()))
    print("Bias_IPW1: " + str(ate_list_LR_linear.mean() - 0))
    print("RMSE_IPW1: " + str(est_rmse(ate_list_LR_linear)))
    print("MAE_IPW1: " + str(np.abs(ate_list_LR_linear).mean()))
    print("Bias_IPW2: " + str(ate2_list_LR_linear.mean() - 0))
    print("RMSE_IPW2: " + str(est_rmse(ate2_list_LR_linear)))
    print("MAE_IPW2: " + str(np.abs(ate2_list_LR_linear).mean()))
    print("RMSE-PS: " + str(rmse_ps_LR_linear.mean()))
    print("MAE-PS: " + str(mae_ps_LR_linear.mean()))
    print("kl-PS: " + str(kldiv_list.mean()))
    
    print(f"Inner count: {inner_count_LR_list.mean()}, Inner RMSE: {inner_rmse_LR_list.mean()}")
    print(f"Outer count: {outer_count_LR_list.mean()}, Outer RMSE: {outer_rmse_LR_list.mean()}")
    print(f"Equal to 0.5, Greater count: {equal_half_greater_count_LR_list.mean()}, Greater RMSE: {equal_half_greater_rmse_LR_list.mean()}")
    print(f"Equal to 0.5, Smaller count: {equal_half_smaller_count_LR_list.mean()}, Smaller RMSE: {equal_half_smaller_rmse_LR_list.mean()}")
    
    # print("GBN")
    # # print("ate_min: " + str(ate_list_GBN_linear.min()))
    # # print("ate_max: " + str(ate_list_GBN_linear.max()))
    # # print("ate_std: " + str(ate_list_GBN_linear.std()))
    # print("Bias_IPW1: " + str(ate_list_GBN_linear.mean() - 0))
    # print("RMSE_IPW1: " + str(est_rmse(ate_list_GBN_linear)))
    # print("MAE_IPW1: " + str(np.abs(ate_list_GBN_linear).mean()))
    # print("Bias_IPW2: " + str(ate2_list_GBN_linear.mean() - 0))
    # print("RMSE_IPW2: " + str(est_rmse(ate2_list_GBN_linear)))
    # print("MAE_IPW2: " + str(np.abs(ate2_list_GBN_linear).mean()))
    # print("RMSE-PS: " + str(rmse_ps_GBN_linear.mean()))
    # print("MAE-PS: " + str(mae_ps_GBN_linear.mean()))
    # print("NCPMIN")
    # # print("ate_min: " + str(ate_list_NCPMIN_linear.min()))
    # # print("ate_max: " + str(ate_list_NCPMIN_linear.max()))
    # # print("ate_std: " + str(ate_list_NCPMIN_linear.std()))
    # print("Bias_IPW1: " + str(ate_list_NCPMIN_linear.mean() - 0))
    # print("RMSE_IPW1: " + str(est_rmse(ate_list_NCPMIN_linear)))
    # print("MAE_IPW1: " + str(np.abs(ate_list_NCPMIN_linear).mean()))
    # print("Bias_IPW2: " + str(ate2_list_NCPMIN_linear.mean() - 0))
    # print("RMSE_IPW2: " + str(est_rmse(ate2_list_NCPMIN_linear)))
    # print("MAE_IPW2: " + str(np.abs(ate2_list_NCPMIN_linear).mean()))
    # print("RMSE-PS: " + str(rmse_ps_NCPMIN_linear.mean()))
    # print("MAE-PS: " + str(mae_ps_NCPMIN_linear.mean()))
    # # print("RMSE-NN: " + str(rmse_ps_LR_linear.mean()))
    # # print("MAE-NN: " + str(mae_ps_LR_linear.mean()))

    # print("nonlinear")
    # print("LR-NN")
    # print("ate_min: " + str(ate_list_LR_nonlinear.min()))
    # print("ate_max: " + str(ate_list_LR_nonlinear.max()))
    # print("E[ate]: " + str(ate_list_LR_nonlinear.mean()))
    # # print("ate_std: " + str(ate_list_LR_nonlinear.std()))
    # print("Bias_IPW1: " + str(ate_list_LR_nonlinear.mean() - 0))
    # print("RMSE_IPW1: " + str(est_rmse(ate_list_LR_nonlinear)))
    # print("MAE_IPW1: " + str(np.abs(ate_list_LR_nonlinear).mean()))
    # print("Bias_IPW2: " + str(ate2_list_LR_nonlinear.mean() - 0))
    # print("RMSE_IPW2: " + str(est_rmse(ate2_list_LR_nonlinear)))
    # print("MAE_IPW2: " + str(np.abs(ate2_list_LR_nonlinear).mean()))
    # print("RMSE-PS: " + str(rmse_ps_LR_nonlinear.mean()))
    # print("MAE-PS: " + str(mae_ps_LR_nonlinear.mean()))
    # # # test.test()
    # print("GBN")
    # # print("ate_min: " + str(ate_list_GBN_nonlinear.min()))
    # # print("ate_max: " + str(ate_list_GBN_nonlinear.max()))
    # # print("ate_std: " + str(ate_list_GBN_nonlinear.std()))
    # print("Bias_IPW1: " + str(ate_list_GBN_nonlinear.mean() - 0))
    # print("RMSE_IPW1: " + str(est_rmse(ate_list_GBN_nonlinear)))
    # print("MAE_IPW1: " + str(np.abs(ate_list_GBN_nonlinear).mean()))
    # print("Bias_IPW2: " + str(ate2_list_GBN_nonlinear.mean() - 0))
    # print("RMSE_IPW2: " + str(est_rmse(ate2_list_GBN_nonlinear)))
    # print("MAE_IPW2: " + str(np.abs(ate2_list_GBN_nonlinear).mean()))
    # print("RMSE-PS: " + str(rmse_ps_GBN_nonlinear.mean()))
    # print("MAE-PS: " + str(mae_ps_GBN_nonlinear.mean()))
    # print("NCPMIN")
    # # print("ate_min: " + str(ate_list_NCPMIN_nonlinear.min()))
    # # print("ate_max: " + str(ate_list_NCPMIN_nonlinear.max()))
    # # print("ate_std: " + str(ate_list_NCPMIN_nonlinear.std()))
    # print("Bias_IPW1: " + str(ate_list_NCPMIN_nonlinear.mean() - 0))
    # print("RMSE_IPW1: " + str(est_rmse(ate_list_NCPMIN_nonlinear)))
    # print("MAE_IPW1: " + str(np.abs(ate_list_NCPMIN_nonlinear).mean()))
    # print("Bias_IPW2: " + str(ate2_list_NCPMIN_nonlinear.mean() - 0))
    # print("RMSE_IPW2: " + str(est_rmse(ate2_list_NCPMIN_nonlinear)))
    # print("MAE_IPW2: " + str(np.abs(ate2_list_NCPMIN_nonlinear).mean()))
    # print("RMSE-PS: " + str(rmse_ps_NCPMIN_nonlinear.mean()))
    # print("MAE-PS: " + str(mae_ps_NCPMIN_nonlinear.mean()))

    
    # print("RMSE-NN: " + str(rmse_ps_LR_nonlinear.mean()))
    # print("MAE-NN: " + str(mae_ps_LR_nonlinear.mean()))
    # test.test()

    
    


if __name__ == "__main__":
    args = sys.argv
    if 2 <= len(args):
        main(args)
    else:
        print("Arguments are too short: 1st position is samplesize.")
        sys.exit(1)