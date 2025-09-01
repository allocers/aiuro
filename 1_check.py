# Imports
import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

# Ignore pandas warnings
warnings.filterwarnings("ignore")

# --- TODO: S3からデータを取得 for intern製
BASE_DIR = os.path.dirname(os.path.abspath('__file__')) # この.ipynbのある場所
CODE_DIR = os.path.join(BASE_DIR, "code")

# codeをsys.pathに追加
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

# Import custom libraries
import importlib
import estimators, estimators_2, data
from data import *
# from estimators import compute_ocd, estimate_weighted_increments, get_rorr_estimates
import estimators as est
import estimators_2 as est2
from plot import plot_simulation

# Reload custom libraries
importlib.reload(estimators)
importlib.reload(estimators_2)
importlib.reload(data)

# --- Simulation setup
NUM_STRATA = 2
SEED = 1024
SAMPLE_SIZES = [10000, 100000, 1000000] # Not used in the shown cells, but likely for a loop

# --- Data simulation and exploration
N = 50000
df = data.simulate_dataset(N, NUM_STRATA)
print(df.head())
print("t counts:")
print(df.t.value_counts().sort_index())
print("x counts:")
print(df.x.value_counts().sort_index())

# --- Model fitting (Logistic Regression for pscore_model, RandomForest for outcome_model)
pscore_model = LogisticRegression()
pscore_model.fit(df.x.values.reshape(-1, 1), df.t) # xを説明変数にするなら、df[[1]]にする

outcome_model = RandomForestRegressor(n_estimators=50, random_state=SEED)
X_aug = np.column_stack([df.x, df.t]) # 交互作用があるならdf[['x', 't']]で列挙
outcome_model.fit(X_aug, df.y)

# --- Estimation using custom functions
aipw_matrix = est2.estimate_aipw_matrix(df, outcome_model, pscore_model)
estimate, se, ci = est2.estimate_weighted_increments(df, outcome_model, pscore_model)
aie_target = df[df.t < df.t.max()].incremental.mean()
results_aie = []
results_aie.append([N, estimate, f"({ci[0]:.3f}, {ci[1]:.3f})", aie_target])
columns_aie = ["Sample Size", "Empirical AIE", "AIE CI", "AIE Target"]
print(results_aie[:5])  # 先頭5行だけ

df_aie = pd.DataFrame(results_aie, columns=columns_aie)
print(df_aie)
g_model=RandomForestRegressor(n_estimators=50,random_state=42).fit(df[['x']],df['y'])
h_model=LinearRegression().fit(df[['x']],df['t'])
# --- RORR Estimation (Regression with One-to-one Regression)
coef, se, ci = est2.get_rorr_estimates(df, g_model, h_model) # g_modelはlinear
rorr_target = (df.assign(plm_plim=lambda df: df.weight / (df.t_star + 1))).plm_plim.mean()
results_rorr = []
results_rorr.append([N, coef, f"({ci[0]:.3f}, {ci[1]:.3f})", rorr_target])
columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI", "RORR Target"]
df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
print(df_rorr)
#ここまではできた

# --- Cross-fitted AIPW function
# Cross-fit (tに偏りがあると動かない)
def crossfit_aipw(df, outcome_model_class, pscore_model_class, K=5, random_state=42):
    """
    Cross-fitted AIPW estimator with K-fold splitting.
    """
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    all_influences = []
    
    # Calculate weights based on t values
    t_vals, counts = np.unique(df['t'], return_counts=True)
    max_t = np.max(t_vals)
    print(max_t)
    weights = np.zeros(max_t + 1)
    weights[t_vals] = counts / len(df)
    
    for train_idx, test_idx in kf.split(df, df['t']):
        # Split data
        df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
        
        # Instantiate and fit pscore and outcome models
        pscore_model = pscore_model_class()
        pscore_model.fit(df_train.x.values.reshape(-1, 1), df_train.t)
        print(pscore_model.classes_)
        outcome_model = outcome_model_class()
        X_aug_train = np.column_stack([df_train.x, df_train.t])
        outcome_model.fit(X_aug_train, df_train.y)
        
        # Estimate AIPW matrix and append
        aipw_matrix = est2.estimate_aipw_matrix(df_test, outcome_model, pscore_model)
        all_influences.append(aipw_matrix)
    
    # Concatenate all influences
    aipw_matrix_full = pd.concat(all_influences, axis=0).sort_index()

    # Compute consecutive mean differences (this part is not in the crossfit_aipw function)
    # The image shows this code block separate from the function definition
    # So, we'll implement it as a separate step after the function call
    
    # Compute the c vector
    T_max = aipw_matrix_full.shape[1] - 1
    c = np.zeros(T_max + 1)
    c[0] = -weights[0]
    c[1:] = weights[:-1] - weights[1:]
    
    # The code below is shown as a standalone block in the images,
    # but based on the context of 'return', it seems to be part of a larger function,
    # which is not fully shown. For completeness, I'll include the calculations.
    
    # Estimate and variance calculation
    psi = aipw_matrix_full @ c
    estimate = np.mean(psi)
    var_hat = np.var(psi, ddof=1) / len(df)
    se = np.sqrt(var_hat)
    ci = [estimate - 1.96 * se, estimate + 1.96 * se]
    
    return estimate, se, ci, aipw_matrix_full

# --- Run cross-fitted AIPW
estimate_crossfit, se_crossfit, ci_crossfit, aipw_matrix_full = crossfit_aipw(
    df,
    outcome_model_class=lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    pscore_model_class=LogisticRegression
)

# --- Display results
aie_target_cf = df[df.t < df.t.max()].incremental.mean()
results_aie_cf = []
results_aie_cf.append([N, estimate_crossfit, f"({ci_crossfit[0]:.3f},{ci_crossfit[1]:.3f})", aie_target_cf])
columns_aie_cf = ["Sample Size", "Empirical AIE", "AIE CI", "AIE Target"]
df_aie_cf = pd.DataFrame(results_aie_cf, columns=columns_aie_cf)
print(df_aie_cf)