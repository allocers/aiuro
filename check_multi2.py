import sys
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: 以下のコードは、ご提示いただいたコードで使われている
# 'estimators_2.py' と 'data_multi.py' が存在し、
# 適切にインポートできることを前提としています。
# ここでは、それらのファイルが存在すると仮定して処理を進めます。

# --- Assume `estimators_2` and `data_multi` are available ---
# For demonstration, we create dummy modules and functions
class DummyModel:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))
    def predict_proba(self, X): return np.ones((len(X), np.max(X.iloc[:, -1]) + 1)) / (np.max(X.iloc[:, -1]) + 1)

def get_rorr_estimates_dummy(df, g, h):
    return 0.1, 0.01, [0.08, 0.12]

def estimate_aipw_matrix_dummy(df, o, p, t_max):
    return pd.DataFrame(np.random.rand(len(df), t_max + 1))

def simulate_dataset_dummy(N, num_cov, num_strata, seed):
    covariates = {f'x{i+1}': np.random.rand(N) for i in range(num_cov)}
    df = pd.DataFrame(covariates)
    df['t'] = np.random.randint(0, 5, N)
    df['y'] = np.random.rand(N)
    df['plm_plim'] = 0.1 # RORR target
    df['incremental'] = 0.05 # AIE target
    return df

# Replace with actual modules if available
try:
    import estimators_2 as est2
    import data_multi as data
except ImportError:
    print("Warning: 'estimators_2' and 'data_multi' not found. Using dummy functions.")
    est2 = type('est', (), {'get_rorr_estimates': get_rorr_estimates_dummy, 'estimate_aipw_matrix': estimate_aipw_matrix_dummy})
    data = type('data', (), {'simulate_dataset': simulate_dataset_dummy, 'NUM_COVARIATES': 5})

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# --- Main simulation function ---
def run_simulation(N, seed=1024):
    """
    指定されたサンプルサイズNでRORRとAIPWの推定を実行する関数
    """
    # --- Data simulation ---
    NUM_COVARIATES = data.NUM_COVARIATES
    NUM_STRATA = 2
    df = data.simulate_dataset(N, NUM_COVARIATES, NUM_STRATA, seed=seed)
    covariate_cols = [f'x{i+1}' for i in range(NUM_COVARIATES)]

    # --- RORR Estimation ---
    g_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(df[covariate_cols], df['y'])
    h_model = LinearRegression().fit(df[covariate_cols], df['t'])
    coef, _, ci_rorr = est2.get_rorr_estimates(df, g_model, h_model)
    rorr_target = df['plm_plim'].mean()

    # --- Cross-fitted AIPW Estimation ---
    def crossfit_aipw(df_in, outcome_model_class, pscore_model_class, K=5, random_state=42):
        df = df_in.copy()
        t_counts = df['t'].value_counts()
        if K > t_counts.min(): K = max(2, t_counts.min())
            
        kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
        all_influences = []
        t_vals, counts = np.unique(df['t'], return_counts=True)
        max_t = np.max(t_vals)
        weights = np.zeros(max_t + 1); weights[t_vals] = counts / len(df)
        
        for train_idx, test_idx in kf.split(df, df['t']):
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
            pscore_model = pscore_model_class(max_iter=1000).fit(df_train[covariate_cols], df_train.t)
            outcome_model = outcome_model_class().fit(pd.concat([df_train[covariate_cols], df_train['t']], axis=1), df_train.y)
            aipw_matrix = est2.estimate_aipw_matrix(df_test, outcome_model, pscore_model, t_max=max_t)
            all_influences.append(aipw_matrix)
        
        aipw_matrix_full = pd.concat(all_influences, axis=0).sort_index()
        w_shifted = np.roll(weights, 1); w_shifted[0] = 0
        c = w_shifted - weights
        psi = (aipw_matrix_full @ c).to_numpy()
        estimate = np.mean(psi); var_hat = np.var(psi, ddof=1) / len(df)
        se = np.sqrt(var_hat); ci = [estimate - 1.96 * se, estimate + 1.96 * se]
        return estimate, ci

    estimate_crossfit, ci_aipw = crossfit_aipw(
        df,
        outcome_model_class=lambda: RandomForestRegressor(n_estimators=50, random_state=42),
        pscore_model_class=LogisticRegression
    )
    aie_target_cf = df[df.t < df.t.max()].incremental.mean()

    return {
        "RORR_Estimate": coef, "RORR_CI": ci_rorr, "RORR_Target": rorr_target,
        "AIPW_Estimate": estimate_crossfit, "AIPW_CI": ci_aipw, "AIPW_Target": aie_target_cf
    }

# --- 1. サンプルサイズとバイアスの関係を評価 ---
sample_sizes = [5000, 10000, 20000, 50000]
results_list = []
for n in sample_sizes:
    print(f"--- Running simulation for N = {n} ---")
    res = run_simulation(n)
    rorr_bias = abs(res["RORR_Estimate"] - res["RORR_Target"])
    aipw_bias = abs(res["AIPW_Estimate"] - res["AIPW_Target"])
    results_list.append({"Sample Size": n, "Estimator": "RORR", "Bias": rorr_bias})
    results_list.append({"Sample Size": n, "Estimator": "AIPW (Cross-fit)", "Bias": aipw_bias})

bias_df = pd.DataFrame(results_list)

# --- 2. 最終結果（最大サンプルサイズ）を評価 ---
final_results = run_simulation(sample_sizes[-1])
plot_data = [
    {"Estimator": "RORR", "Value": final_results["RORR_Estimate"], "CI_low": final_results["RORR_CI"][0], "CI_high": final_results["RORR_CI"][1], "Target": final_results["RORR_Target"]},
    {"Estimator": "AIPW (Cross-fit)", "Value": final_results["AIPW_Estimate"], "CI_low": final_results["AIPW_CI"][0], "CI_high": final_results["AIPW_CI"][1], "Target": final_results["AIPW_Target"]},
]
final_df = pd.DataFrame(plot_data)

# --- グラフ描画 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# グラフ1: サンプルサイズ vs バイアス
sns.lineplot(data=bias_df, x="Sample Size", y="Bias", hue="Estimator",
             style="Estimator", markers=True, dashes=False, ax=axes[0], palette="viridis", lw=2.5)
axes[0].set_title("Impact of Sample Size on Estimator Bias", fontsize=16, pad=20)
axes[0].set_xlabel("Sample Size (N)", fontsize=12)
axes[0].set_ylabel("Absolute Bias |Estimate - Target|", fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend(title='Estimator')

# グラフ2: 推定結果の比較
y_err = [final_df['Value'] - final_df['CI_low'], final_df['CI_high'] - final_df['Value']]
colors = sns.color_palette("viridis", n_colors=len(final_df))
axes[1].bar(final_df['Estimator'], final_df['Value'], yerr=y_err, capsize=5, color=colors, alpha=0.8)

for i, row in final_df.iterrows():
    axes[1].hlines(row['Target'], xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='dashed', label='Target Value' if i == 0 else "")

axes[1].set_title(f"Estimation Results at N={sample_sizes[-1]}", fontsize=16, pad=20)
axes[1].set_ylabel("Estimated Value", fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n--- Bias Results ---")
print(bias_df)
print("\n--- Final Estimation Results ---")
print(final_df)