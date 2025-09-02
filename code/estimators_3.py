import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

def get_rorr_estimates_dummy(df, g, h):
    """
    ダミーのRORR推定値を返す。
    """
    return 0.1, 0.01, [0.08, 0.12]

def estimate_aipw_matrix_dummy(df, o, p, t_max):
    """
    ダミーのAIPW行列を返す。
    """
    return pd.DataFrame(np.random.rand(len(df), t_max + 1))

def simulate_dataset(N, num_cov, num_strata, seed):
    """
    シミュレートされたデータセットを生成する。
    """
    np.random.seed(seed)
    covariates = {f'x{i+1}': np.random.rand(N) for i in range(num_cov)}
    df = pd.DataFrame(covariates)
    
    # 処置tは共変量に依存するように設定
    p_t = 1 / (1 + np.exp(-(df['x1'] + df['x2'])))
    df['t'] = np.random.choice([0, 1], size=N, p=[1 - p_t, p_t])

    # 処置前のアウトカムも共変量に依存するように設定
    df['y_pre'] = df['x1'] * 0.5 + df['x2'] * 0.3 + np.random.randn(N) * 0.5

    # アウトカムyは処置と共変量に依存するように設定
    df['y'] = df['t'] * 0.5 + df['x1'] * 0.8 + df['x2'] * 0.5 + np.random.randn(N)
    
    # Stratum/Binning for the plot
    df['strata'] = pd.qcut(df['y_pre'], num_strata, labels=False)
    
    return df

def get_aipw_matrix(df, outcome_model, pscore_model, t_max):
    """
    AIPW行列を計算する。
    """
    df = df.copy()
    covariates = [c for c in df.columns if c.startswith('x')]
    
    # Estimate outcome model Q(t, x)
    df_with_t = df[covariates + ['t']]
    df['q_pred'] = outcome_model.predict(df_with_t)

    # Estimate propensity score model e(x) = P(t | x)
    # The `predict_proba` method of sklearn models is used to get p-scores
    # It returns an array where each column corresponds to a class/treatment value.
    # The columns are ordered by the class labels found in pscore_model.classes_
    pscore_proba = pscore_model.predict_proba(df[covariates])
    pscore_df = pd.DataFrame(pscore_proba, columns=pscore_model.classes_)

    # Correctly map the probabilities back to the original treatment values
    df['pscore'] = pscore_df.lookup(df.index, df['t'])

    aipw_matrix = pd.DataFrame()
    for t_val in range(t_max + 1):
        # Create a counterfactual dataframe
        df_counterfactual = df[covariates].copy()
        df_counterfactual['t'] = t_val
        q_counterfactual = outcome_model.predict(df_counterfactual)
        
        # Calculate the AIPW component for each treatment level
        aipw_t = q_counterfactual + (df['y'] - df['q_pred']) * (df['t'] == t_val) / df['pscore']
        aipw_matrix[t_val] = aipw_t
        
    return aipw_matrix

def check_balance(df, pscore_model):
    """
    IPWを用いて、処置前のアウトカムのバランスを評価する。
    """
    covariate_cols = [c for c in df.columns if c.startswith('x')]
    
    # 傾向スコアの計算
    df['propensity_score'] = pscore_model.predict_proba(df[covariate_cols])[:, 1]
    
    # IPWの計算
    df['ipw'] = df.apply(lambda row: 1 / row['propensity_score'] if row['t'] == 1 else 1 / (1 - row['propensity_score']), axis=1)

    # 標準化された平均差 (Standardized Mean Difference) を計算
    def get_smd(df_subset, weights=None):
        if weights is None:
            weights = np.ones(len(df_subset))
        
        # Treatment and control groups
        df_t = df_subset[df_subset['t'] == 1]
        df_c = df_subset[df_subset['t'] == 0]
        
        # Weighted means and variances
        mean_t = np.average(df_t['y_pre'], weights=weights[df_t.index])
        mean_c = np.average(df_c['y_pre'], weights=weights[df_c.index])
        
        var_t = np.average((df_t['y_pre'] - mean_t)**2, weights=weights[df_t.index])
        var_c = np.average((df_c['y_pre'] - mean_c)**2, weights=weights[df_c.index])
        
        # Pooled standard deviation
        sd_pool = np.sqrt((var_t + var_c) / 2)
        
        if sd_pool == 0:
            return 0
        return (mean_t - mean_c) / sd_pool

    # 重み付け前のSMD
    smd_before = get_smd(df)
    # 重み付け後のSMD
    smd_after = get_smd(df, df['ipw'])
    
    # バランスのプロット
    # Stratified SMD for the plot
    strata = sorted(df['strata'].unique())
    smds_before = [get_smd(df[df['strata'] == s]) for s in strata]
    smds_after = [get_smd(df[df['strata'] == s], df[df['strata'] == s]['ipw']) for s in strata]
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(smds_before, strata, c='orange', s=100, label='Before IPW')
    plt.scatter(smds_after, strata, c='blue', s=100, label='After IPW')
    
    plt.axvline(x=0, color='red', linestyle='-', linewidth=1)
    plt.axvline(x=-0.1, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0.1, color='gray', linestyle='--', linewidth=1)
    
    plt.yticks(strata)
    plt.xlabel('Standardized Difference in Mean Pre-Treatment Visits')
    plt.ylabel('Feature A Usage Bins')
    plt.title('Balance in Pre-Treatment Outcome After Weighting')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.show()

    return smd_before, smd_after, df['ipw']
