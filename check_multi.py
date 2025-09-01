import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# --- TODO: S3からデータを取得 for intern製
BASE_DIR = os.path.dirname(os.path.abspath('__file__')) # この.ipynbのある場所
CODE_DIR = os.path.join(BASE_DIR, "code")

# codeをsys.pathに追加
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

import estimators_2 as est2
import data_multi as data
import importlib

importlib.reload(est2)
importlib.reload(data)

# --- Simulation setup
NUM_STRATA = 2
SEED = 1024
N = 50000

# --- Data simulation
NUM_COVARIATES = data.NUM_COVARIATES
df = data.simulate_dataset(N, NUM_COVARIATES, NUM_STRATA, seed=SEED)
covariate_cols = [f'x{i+1}' for i in range(NUM_COVARIATES)]
print("--- Data Head ---")
print(df.head())
print("t counts:")
print(df.t.value_counts().sort_index())
print("x1 counts:")
print(df.x1.value_counts().sort_index())

# --- RORR Estimation
print("\n--- Running RORR Estimation ---")
g_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(df[covariate_cols], df['y'])
h_model = LinearRegression().fit(df[covariate_cols], df['t'])

coef, se, ci = est2.get_rorr_estimates(df, g_model, h_model)
rorr_target = df['plm_plim'].mean()
results_rorr = [[N, coef, f"({ci[0]:.3f}, {ci[1]:.3f})", rorr_target]]
columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI", "RORR Target"]
df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
print(df_rorr)

# --- Cross-fitted AIPW function
def crossfit_aipw(df, outcome_model_class, pscore_model_class, K=5, random_state=42):
    t_counts = df['t'].value_counts()
    valid_t = t_counts[t_counts >= K].index
    df_filtered = df[df['t'].isin(valid_t)].copy()
    
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    all_influences = []
    
    t_vals, counts = np.unique(df['t'], return_counts=True)
    max_t = np.max(t_vals)
    weights = np.zeros(max_t + 1)
    weights[t_vals] = counts / len(df)
    
    covariate_cols = [col for col in df.columns if col.startswith('x')]
    
    print(f"Original size: {len(df)}, Filtered size for cross-fitting: {len(df_filtered)}")

    for train_idx, test_idx in kf.split(df_filtered, df_filtered['t']):
        df_train, df_test = df_filtered.iloc[train_idx], df_filtered.iloc[test_idx]
        
        pscore_model = pscore_model_class(max_iter=1000)
        pscore_model.fit(df_train[covariate_cols], df_train.t)
        
        outcome_model = outcome_model_class()
        X_aug_train = pd.concat([df_train[covariate_cols], df_train['t']], axis=1)
        outcome_model.fit(X_aug_train, df_train.y)
        
        aipw_matrix = est2.estimate_aipw_matrix(df_test, outcome_model, pscore_model, t_max=max_t)
        all_influences.append(aipw_matrix)
    
    aipw_matrix_full = pd.concat(all_influences, axis=0).sort_index()

    T_max = aipw_matrix_full.shape[1] - 1
    c = np.zeros(T_max + 1)
    
    w_shifted = np.roll(weights, 1)
    w_shifted[0] = 0
    c = w_shifted - weights
    
    psi = (aipw_matrix_full @ c).to_numpy()
    estimate = np.mean(psi)
    var_hat = np.var(psi, ddof=1) / len(df_filtered)
    se = np.sqrt(var_hat)
    ci = [estimate - 1.96 * se, estimate + 1.96 * se]
    
    return estimate, se, ci

# --- Run cross-fitted AIPW
print("\n--- Running Cross-Fitted AIPW Estimation ---")
estimate_crossfit, se_crossfit, ci_crossfit = crossfit_aipw(
    df,
    outcome_model_class=lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    pscore_model_class=LogisticRegression
)

aie_target_cf = df[df.t < df.t.max()].incremental.mean()
results_aie_cf = [[N, estimate_crossfit, f"({ci_crossfit[0]:.3f},{ci_crossfit[1]:.3f})", aie_target_cf]]
columns_aie_cf = ["Sample Size", "Empirical AIE (Cross-fit)", "AIE CI", "AIE Target"]
df_aie_cf = pd.DataFrame(results_aie_cf, columns=columns_aie_cf)
print(df_aie_cf)