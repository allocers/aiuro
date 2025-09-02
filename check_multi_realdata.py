import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
# estimators_2.py が同じディレクトリにあることを確認してください
# import estimators_2 as est2
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
warnings.filterwarnings("ignore")
# Replace with actual modules if available
try:
    import estimators_2 as est2
    import data_multi as data
except ImportError:
    print("Warning: 'estimators_2' and 'data_multi' not found. Using dummy functions.")
    est2 = type('est', (), {'get_rorr_estimates': get_rorr_estimates_dummy, 'estimate_aipw_matrix': estimate_aipw_matrix_dummy})
    data = type('data', (), {'simulate_dataset': simulate_dataset_dummy, 'NUM_COVARIATES': 5})

# --- 1. データ読み込み ---
# この部分を実際のデータ読み込みに置き換えてください。
# 例: df = pd.read_csv('your_data.csv')

# 以下は、コードがすぐに実行できるようにするためのダミーデータです。
# 実際のデータを使う際は、この部分をコメントアウトまたは削除してください。
print("--- Creating Dummy Data for Demonstration ---")
N = 10000  # サンプルサイズ
dummy_data = {
    't': np.random.randint(0, 5, size=N),
    'y': np.random.randn(N),
    'x1': np.random.randn(N),
    'x2': np.random.binomial(1, 0.4, size=N),
    'x3': np.random.randn(N),
}
df = pd.DataFrame(dummy_data)

print("--- Data Head ---")
print(df.head())

# --- 2. 変数定義 ---
# 共変量（特徴量）のカラム名を指定
covariate_cols = ['x1', 'x2', 'x3']
N = len(df) # サンプルサイズをデータから取得

# --- 3. RORR (Ratio of Ratios) 推定 ---
print("\n--- Running RORR Estimation ---")
# アウトカムモデル y ~ x1 + x2 + x3
g_model = RandomForestRegressor(n_estimators=50, random_state=42).fit(df[covariate_cols], df['y'])
# 介入モデル t ~ x1 + x2 + x3
h_model = LinearRegression().fit(df[covariate_cols], df['t'])

# RORRの推定値を計算
coef, se, ci = est2.get_rorr_estimates(df, g_model, h_model)

# 結果の表示（Target列は削除）
results_rorr = [[N, coef, f"({ci[0]:.3f}, {ci[1]:.3f})"]]
columns_rorr = ["Sample Size", "Empirical RORR", "RORR CI"]
df_rorr = pd.DataFrame(results_rorr, columns=columns_rorr)
print(df_rorr)


# --- 4. Cross-fitted AIPW のための関数定義 ---
def crossfit_aipw(df, covariate_cols, outcome_model_class, pscore_model_class, K=5, random_state=42):
    """
    交差適合（Cross-fitting）を用いてAIPW推定量（AIE: Average Incremental Effect）を計算する関数
    """
    # 介入tの値ごとにK個以上のサンプルがない場合、そのデータは除外
    t_counts = df['t'].value_counts()
    valid_t = t_counts[t_counts >= K].index
    df_filtered = df[df['t'].isin(valid_t)].copy()
    
    kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    all_influences = []
    
    t_vals, counts = np.unique(df['t'], return_counts=True)
    max_t = np.max(t_vals)
    weights = np.zeros(max_t + 1)
    weights[t_vals] = counts / len(df)
    
    print(f"Original size: {len(df)}, Filtered size for cross-fitting: {len(df_filtered)}")

    # K-分割交差検証
    for train_idx, test_idx in kf.split(df_filtered, df_filtered['t']):
        df_train, df_test = df_filtered.iloc[train_idx], df_filtered.iloc[test_idx]
        
        # 傾向スコアモデルの学習（分割した一方のデータで）
        pscore_model = pscore_model_class(max_iter=1000)
        pscore_model.fit(df_train[covariate_cols], df_train.t)
        
        # アウトカムモデルの学習（分割した一方のデータで）
        outcome_model = outcome_model_class()
        # アウトカムモデルは共変量Xと介入Tの両方を使ってyを予測
        X_aug_train = pd.concat([df_train[covariate_cols], df_train['t']], axis=1)
        outcome_model.fit(X_aug_train, df_train.y)
        
        # AIPWの計算（もう一方のデータで）
        aipw_matrix = est2.estimate_aipw_matrix(df_test, outcome_model, pscore_model, t_max=max_t)
        all_influences.append(aipw_matrix)
    
    aipw_matrix_full = pd.concat(all_influences, axis=0).sort_index()

    # AIE (Average Incremental Effect) の計算
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

# --- 5. Cross-fitted AIPW 推定の実行 ---
print("\n--- Running Cross-Fitted AIPW Estimation ---")
# AIPW推定量の計算
estimate_crossfit, se_crossfit, ci_crossfit = crossfit_aipw(
    df,
    covariate_cols, # 共変量を指定
    outcome_model_class=lambda: RandomForestRegressor(n_estimators=50, random_state=42),
    pscore_model_class=LogisticRegression
)

# 結果の表示（Target列は削除）
results_aie_cf = [[N, estimate_crossfit, f"({ci_crossfit[0]:.3f},{ci_crossfit[1]:.3f})"]]
columns_aie_cf = ["Sample Size", "Empirical AIE (Cross-fit)", "AIE CI"]
df_aie_cf = pd.DataFrame(results_aie_cf, columns=columns_aie_cf)
print(df_aie_cf)