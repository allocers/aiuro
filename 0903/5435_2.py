import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
# Read the CSV files
df_control = pd.read_csv('CPID5435_control2.csv')
df_treatment = pd.read_csv('CPID5435_treatment2.csv')

# Add the 't' column
df_control['t'] = 0
df_treatment['t'] = 1

# Concatenate the dataframes
df = pd.concat([df_control, df_treatment], ignore_index=True)

# Define the columns to standardize
covariate_cols = ['DPAY_SETTLEMENT_AMOUNT_202406', 'DPOINTS_USE_202406','DPOINTS_USE_202407','DPOINTS_USE_202408','DPOINTS_USE_202409','DPOINTS_GIVE_202406', 'DPOINTS_GIVE_202407', 'DPOINTS_GIVE_202408', 'DPOINTS_GIVE_202409', 'DCARD_USE_AMOUNT_202406', 'CONTRACT', 'CONTRACT_PERIOD', 'GENDER', 'AGE']
##アウトカム
##CP期間７月なので7月の決済金額-6月の決済金額
df['y'] = df['DPAY_SETTLEMENT_AMOUNT_202407'] - df['DPAY_SETTLEMENT_AMOUNT_202406']
df['GENDER']=np.where(df['GENDER']==1,1,np.where(df['GENDER']==2,0,np.nan))

# Display the top 5 rows
print("結合・標準化後のデータフレームの上位5行:")
print(df.head())


# print(df.info())
# RangeIndex: 2000 entries, 0 to 1999
# Data columns (total 20 columns):

#  #   Column                         Non-Null Count  Dtype
# ---  ------                         --------------  -----
#  0   COMNID                         2000 non-null   int64
#  1   DPAY_SETTLEMENT_AMOUNT_202406  2000 non-null   float64
#  2   DPAY_SETTLEMENT_AMOUNT_202407  2000 non-null   int64
#  3   DPAY_SETTLEMENT_AMOUNT_202408  2000 non-null   int64
#  4   DPAY_SETTLEMENT_AMOUNT_202409  2000 non-null   int64
#  5   DPOINTS_USE_202406             2000 non-null   float64
#  6   DPOINTS_USE_202407             2000 non-null   float64
#  7   DPOINTS_USE_202408             2000 non-null   float64
#  8   DPOINTS_USE_202409             2000 non-null   float64
#  9   DPOINTS_GIVE_202406            2000 non-null   float64
#  10  DPOINTS_GIVE_202407            2000 non-null   float64
#  11  DPOINTS_GIVE_202408            2000 non-null   float64
#  12  DPOINTS_GIVE_202409            2000 non-null   float64
#  13  DCARD_USE_AMOUNT_202406        2000 non-null   float64
#  14  CONTRACT                       2000 non-null   int64
#  15  CONTRACT_PERIOD                2000 non-null   int64
#  16  GENDER                         2000 non-null   int64
#  17  AGE                            2000 non-null   int64
#  18  t                              2000 non-null   int64
#  19  y                              2000 non-null   int64
# dtypes: float64(10), int64(10)
# memory usage: 312.6 KB
# None
# ---
### 共変量の分布を画像として保存

num_covariates = len(covariate_cols)
n_cols = 4  # 4列に設定
n_rows = (num_covariates + n_cols - 1) // n_cols  # 動的に行数を計算

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 16))
fig.suptitle('Covariate Distribution by Treatment Group (Standardized)', fontsize=20)
axes = axes.flatten()

for i, col in enumerate(covariate_cols):
    # CONTRACT, GENDER, CONTRACT_PERIOD の kde を無効化
    if col in ['CONTRACT', 'GENDER', 'CONTRACT_PERIOD']:
        sns.histplot(data=df, x=col, hue='t', ax=axes[i], palette='viridis')
    else:
        sns.histplot(data=df, x=col, hue='t', ax=axes[i], palette='viridis')

# 未使用のサブプロットを非表示にする
for j in range(num_covariates, len(axes)):
    axes[j].set_visible(False)

# レイアウトを調整し、画像として保存
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('covariate_distributions.png')
print("共変量の分布グラフを 'covariate_distributions.png' として保存しました。")
plt.close(fig) # メモリを解放

# ---
### 共変量間の相関を画像として保存

correlation_matrix = df[covariate_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Covariates', fontsize=16)

# 画像として保存
plt.savefig('correlation_matrix.png')
print("共変量間の相関ヒートマップを 'correlation_matrix.png' として保存しました。")
# plt.show()
# --- アウトカムと共変量の相関を計算 ---
# 相関を計算したいカラムのリストを作成
cols_for_corr = covariate_cols + ['y']

# 相関行列を計算
correlation_with_outcome = df[cols_for_corr].corr()

# アウトカム 'y' と各共変量との相関のみを抽出
corr_y = correlation_with_outcome['y'].drop('y') # 自分自身との相関は除外

# 相関の強さでソート
corr_y_sorted = corr_y.sort_values(ascending=False)

print("\n--- アウトカム(y)と共変量の相関係数 ---")
print(corr_y_sorted)


# --- 相関を可視化 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))

# Seabornのbarplotを使用して可視化
sns.barplot(x=corr_y_sorted.values, y=corr_y_sorted.index, palette='coolwarm')

# グラフのタイトルとラベルを設定
plt.title('Correlation between Covariates and Outcome (y)', fontsize=16)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Covariates')
plt.axvline(0, color='black', linewidth=0.8) # ゼロ相関の線を追加
plt.tight_layout()

# 画像として保存
plt.savefig('outcome_correlation.png')
print("\nアウトカムと共変量の相関グラフを 'outcome_correlation.png' として保存しました。")

# グラフを表示
plt.show()

# # Standardize the specified columns
scaler = StandardScaler()
cols_to_standardize = ['DPAY_SETTLEMENT_AMOUNT_202406', 'DPOINTS_USE_202406','DPOINTS_USE_202407','DPOINTS_USE_202408','DPOINTS_USE_202409','DPOINTS_GIVE_202406', 'DPOINTS_GIVE_202407', 'DPOINTS_GIVE_202408', 'DPOINTS_GIVE_202409', 'DCARD_USE_AMOUNT_202406']
df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
# --- チューニング用のパラメータグリッドを定義 ---
param_grid_ps = {'C': [0.1, 1, 10], 'max_iter': [1000]}
param_grid_rf = {'n_estimators': [100], 'max_depth': [10], 'min_samples_leaf': [20]}

# --- 交差検証の準備 ---
n_splits = 10
random_state = 42
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

X = df[covariate_cols].values
T = df['t'].values
Y = df['y'].values

# OOF予測値と評価指標を格納するオブジェクト
df['ps_oof'] = np.nan
df['g_hat_oof'] = np.nan
df['m0_hat_oof'] = np.nan
df['m1_hat_oof'] = np.nan
aucs_foldwise = []
ate_plugin_foldwise = [] # FoldごとのATEを格納

print("--- ハイパーパラメータチューニング付き交差検証を開始します ---")
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, T), 1):
    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = X[tr_idx], X[te_idx], T[tr_idx], T[te_idx], Y[tr_idx], Y[te_idx]

    # 1. 傾向スコアモデル (AUCも計算)
    grid_ps = GridSearchCV(LogisticRegression(random_state=random_state), param_grid_ps, cv=3, scoring='roc_auc')
    grid_ps.fit(X_tr, T_tr)
    ps_te = grid_ps.predict_proba(X_te)[:, 1]
    df.loc[te_idx, 'ps_oof'] = ps_te
    auc = roc_auc_score(T_te, ps_te)
    aucs_foldwise.append(auc)
    print(f"[Fold {fold:02d}] AUC = {auc:.4f}")

    # 2. RORR用結果モデル
    grid_g = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3)
    grid_g.fit(X_tr, Y_tr)
    df.loc[te_idx, 'g_hat_oof'] = grid_g.predict(X_te)

    # 3. AIPW用結果モデル
    grid_m0 = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3)
    grid_m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
    df.loc[te_idx, 'm0_hat_oof'] = grid_m0.predict(X_te)
    grid_m1 = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3)
    grid_m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
    df.loc[te_idx, 'm1_hat_oof'] = grid_m1.predict(X_te)
    
    # 4. FoldごとのATEを計算 (プラグイン推定量)
    ps_te_clipped = np.clip(ps_te, 0.05, 0.95)
    t1_idx, t0_idx = T_te == 1, T_te == 0
    y1_hat = np.sum(Y_te[t1_idx] / ps_te_clipped[t1_idx]) / np.sum(1 / ps_te_clipped[t1_idx])
    y0_hat = np.sum(Y_te[t0_idx] / (1-ps_te_clipped[t0_idx])) / np.sum(1 / (1-ps_te_clipped[t0_idx]))
    ate_plugin_foldwise.append(y1_hat - y0_hat)

print("-" * 30)
print(f" 平均AUC (±標準偏差) = {np.mean(aucs_foldwise):.4f} ± {np.std(aucs_foldwise, ddof=1):.4f}")
print(" 全てのサンプル外予測値の計算が完了しました。")


thr = 0.05 # クリッピングの閾値

# グラフ描画エリアの設定
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# クリッピング前の分布
sns.histplot(data=df, x='ps_oof', hue='t', bins=30, stat='density', common_norm=False, element='step', ax=axes[0])
axes[0].set_title('Propensity Score Distribution (Unclipped)')
axes[0].set_xlabel('Propensity Score')
axes[0].grid(True)

# クリッピング後の分布
ps_clipped = df['ps_oof'].clip(thr, 1 - thr)
sns.histplot(x=ps_clipped, hue=df['t'], bins=30, stat='density', common_norm=False, element='step', ax=axes[1])
axes[1].set_title(f'Propensity Score Distribution (Clipped at {thr})')
axes[1].set_xlabel('Propensity Score (Clipped)')
axes[1].grid(True)

plt.tight_layout()
plt.show()


# --- 推定量を計算する関数群 ---
def estimate_simple_difference(df, t_name='t', y_name='y'):
    y1 = df.loc[df[t_name] == 1, y_name]
    y0 = df.loc[df[t_name] == 0, y_name]
    effect = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    ci = (effect - 1.96 * se, effect + 1.96 * se)
    return effect, ci

def estimate_ipw_plugin(df, t_name='t', y_name='y', thr=0.05):
    ps = df['ps_oof'].clip(thr, 1 - thr)
    t, y = df[t_name], df[y_name]
    w1 = t / ps
    w0 = (1 - t) / (1 - ps)
    E1 = np.sum(y * w1) / np.sum(w1)
    E0 = np.sum(y * w0) / np.sum(w0)
    E11 = y[t == 1].mean()
    E00 = y[t == 0].mean()
    w01 = ((1 - t) * ps) / (1 - ps)
    E01 = np.sum(y * w01) / np.sum(w01)
    w10 = (t * (1 - ps)) / ps
    E10 = np.sum(y * w10) / np.sum(w10)
    return {
        "ATE": E1 - E0, "ATT": E11 - E01, "ATU": E10 - E00, "SIMPLE": E11 - E00
    }

def estimate_wls(df, t_name='t', y_name='y', thr=0.05, kind='ATE'):
    ps = df['ps_oof'].clip(thr, 1-thr)
    t, y = df[t_name], df[y_name]
    if kind == 'ATE': weights = np.where(t == 1, 1/ps, 1/(1-ps))
    elif kind == 'ATT': weights = np.where(t == 1, 1, ps/(1-ps))
    elif kind == 'ATU': weights = np.where(t == 1, (1-ps)/ps, 1)
    model = sm.WLS(y, sm.add_constant(t), weights=weights).fit()
    return model.params.iloc[1], model.conf_int().iloc[1], model.pvalues.iloc[1]

def estimate_ipw_wls(df, t_name='t', y_name='y'):
    ps_clipped = np.clip(df['ps_oof'], 0.05, 0.95)
    weights = np.where(df[t_name] == 1, 1 / ps_clipped, 1 / (1 - ps_clipped))
    X = sm.add_constant(df[t_name])
    model = sm.WLS(df[y_name], X, weights=weights).fit()
    return model.params.iloc[1], model.conf_int().iloc[1], model.pvalues.iloc[1]

def estimate_rorr_standard(df, t_name='t', y_name='y'):
    t_residual = df[t_name] - df['ps_oof']
    y_residual = df[y_name] - df['g_hat_oof']
    X = sm.add_constant(t_residual)
    model = sm.OLS(y_residual, X).fit()
    return model.params.iloc[1], model.conf_int().iloc[1], model.pvalues.iloc[1]
    
def estimate_aipw_standard(df, t_name='t', y_name='y'):
    T, Y, ps = df[t_name], df[y_name], np.clip(df['ps_oof'], 0.01, 0.99)
    m0, m1 = df['m0_hat_oof'], df['m1_hat_oof']
    psi1 = np.mean((T / ps) * (Y - m1) + m1)
    psi0 = np.mean(((1 - T) / (1 - ps)) * (Y - m0) + m0)
    ate = psi1 - psi0
    
    if1 = (T / ps) * (Y - m1) + m1 - psi1
    if0 = ((1 - T) / (1 - ps)) * (Y - m0) + m0 - psi0
    influence_function = if1 - if0
    se = np.sqrt(np.var(influence_function, ddof=1) / len(df))
    
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    z_score = ate / se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return ate, ci, p_value

# --- 推定の実行と結果の表示 ---
ipw_effect, ipw_ci, ipw_p = estimate_ipw_wls(df)
rorr_effect, rorr_ci, rorr_p = estimate_rorr_standard(df)
aipw_effect, aipw_ci, aipw_p = estimate_aipw_standard(df)

print("\n" + "="*50)
print("           因果効果の推定結果サマリー")
print("="*50)
    
print(f"\n▶ IPW (WLS) 推定量")
print(f"  ATE: {ipw_effect:.2f}")
print(f"  95% CI: ({ipw_ci[0]:.2f}, {ipw_ci[1]:.2f})")
print(f"  P-value: {ipw_p:.4g}")

print(f"\n▶ RORR (標準) 推定量")
print(f"  ATE: {rorr_effect:.2f}")
print(f"  95% CI: ({rorr_ci[0]:.2f}, {rorr_ci[1]:.2f})")
print(f"  P-value: {rorr_p:.4g}")
    
print(f"\n▶ AIPW (標準/二重頑健) 推定量 ✨")
print(f"  ATE: {aipw_effect:.2f}")
print(f"  95% CI: ({aipw_ci[0]:.2f}, {aipw_ci[1]:.2f})")
print(f"  P-value: {aipw_p:.4g}")

print("="*50)
# --- 実行 & サマリー表示 ---
simple_effect, simple_ci = estimate_simple_difference(df)
ipw_plugin_effects = estimate_ipw_plugin(df)

wls_ate, wls_ate_ci, wls_ate_p = estimate_wls(df, kind='ATE')
wls_att, wls_att_ci, wls_att_p = estimate_wls(df, kind='ATT')
wls_atu, wls_atu_ci, wls_atu_p = estimate_wls(df, kind='ATU')

fold_ate_mean = np.mean(ate_plugin_foldwise)
fold_ate_range = np.quantile(ate_plugin_foldwise, [0.025, 0.975])

print("\n" + "="*60)
print("           因果効果の推定結果サマリー（詳細版）")
print("="*60)

print(f"[単純比較]      effect = {simple_effect:.4f}, 95% CI = ({simple_ci[0]:.4f}, {simple_ci[1]:.4f})")
print(f"[IPW plugin]    ATE={ipw_plugin_effects['ATE']:.4f}, ATT={ipw_plugin_effects['ATT']:.4f}, ATU={ipw_plugin_effects['ATU']:.4f}")

print("\n--- WLSによる推定 (p値・信頼区間あり) ---")
print(f"[WLS ATE]       effect={wls_ate:.4f}, p={wls_ate_p:.4g}, 95% CI=({wls_ate_ci[0]:.4f}, {wls_ate_ci[1]:.4f})")
print(f"[WLS ATT]       effect={wls_att:.4f}, p={wls_att_p:.4g}, 95% CI=({wls_att_ci[0]:.4f}, {wls_att_ci[1]:.4f})")
print(f"[WLS ATU]       effect={wls_atu:.4f}, p={wls_atu_p:.4g}, 95% CI=({wls_atu_ci[0]:.4f}, {wls_atu_ci[1]:.4f})")

print("\n--- 参考: Foldごとのばらつき ---")
print(f"[Fold別ATE]     mean={fold_ate_mean:.4f}, 95% range=({fold_ate_range[0]:.4f}, {fold_ate_range[1]:.4f})")
print("="*60)
# --- SMD計算用の関数を定義 ---
def compute_smd(df, covariates, treatment_col, weights=None):
    """指定された重みでSMDを計算する"""
    smd_list = []
    
    # 介入群と対照群のインデックスを取得
    treat_idx = df[treatment_col] == 1
    control_idx = df[treatment_col] == 0
    
    for cov in covariates:
        # 介入群と対照群のデータを抽出
        treat_data = df.loc[treat_idx, cov]
        control_data = df.loc[control_idx, cov]
        
        if weights is None:
            # 重みなしの場合
            mean_treat = treat_data.mean()
            mean_control = control_data.mean()
            var_treat = treat_data.var(ddof=1)
            var_control = control_data.var(ddof=1)
        else:
            # 重みありの場合
            w_treat = weights[treat_idx]
            w_control = weights[control_idx]
            
            mean_treat = np.average(treat_data, weights=w_treat)
            mean_control = np.average(control_data, weights=w_control)
            var_treat = np.average((treat_data - mean_treat)**2, weights=w_treat)
            var_control = np.average((control_data - mean_control)**2, weights=w_control)
            
        # SMDの計算
        pooled_std = np.sqrt((var_treat + var_control) / 2)
        smd = (mean_treat - mean_control) / pooled_std
        smd_list.append(smd)
        
    return smd_list

# --- SMDの計算を実行 ---
# 1. 重み付けなし (補正前)
smd_unweighted = compute_smd(df, covariate_cols, 't')

# 2. IPWによる重み付けあり (補正後)
ps_clipped = np.clip(df['ps_oof'], 0.05, 0.95)
ipw_weights = np.where(df['t'] == 1, 1 / ps_clipped, 1 / (1 - ps_clipped))
smd_weighted = compute_smd(df, covariate_cols, 't', weights=ipw_weights)

# --- 結果をDataFrameにまとめる ---
smd_df = pd.DataFrame({
    'Unweighted': smd_unweighted,
    'IPW Weighted': smd_weighted
}, index=covariate_cols)


# --- SMDの可視化 (Loveプロット) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# ポイントをプロット
ax.scatter(smd_df['Unweighted'], smd_df.index, color='orange', label='Unweighted', alpha=0.8)
ax.scatter(smd_df['IPW Weighted'], smd_df.index, color='blue', label='IPW Weighted', alpha=0.8)

# 基準となる垂直線を追加
ax.axvline(0, color='grey', linestyle='-')
ax.axvline(0.1, color='black', linestyle='--', label='SMD=0.1 基準')
ax.axvline(-0.1, color='black', linestyle='--')

# グラフの体裁を整える
ax.set_xlabel("Standardized Mean Difference (SMD)")
ax.set_ylabel("Covariates")
ax.set_title("Covariate Balance Before & After IPW Weighting")
ax.legend()
plt.tight_layout()
plt.show()
# # --- チューニング用のパラメータグリッドを定義 ---
# param_grid_ps = {
#     'C': [0.1, 1, 10],
#     'penalty': ['l2'],
#     'solver': ['lbfgs'],
#     'max_iter': [1000]
# }
# param_grid_rf = {
#     'n_estimators': [50, 100],
#     'max_depth': [5, 10, None],
#     'min_samples_leaf': [10, 20]
# }

# # --- 交差検証の準備 ---
# n_splits = 10
# random_state = 42
# skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# X = df[covariate_cols].values
# T = df['t'].values
# Y = df['y'].values

# # OOF予測値を格納する列を初期化
# df['ps_oof'] = np.nan       # 傾向スコア P(T=1|X)
# df['g_hat_oof'] = np.nan    # RORR用結果モデル E[Y|X]
# df['m0_hat_oof'] = np.nan   # AIPW用結果モデル E[Y|T=0,X]
# df['m1_hat_oof'] = np.nan   # AIPW用結果モデル E[Y|T=1,X]

# print("--- ハイパーパラメータチューニング付き交差検証を開始します ---")
# for fold, (tr_idx, te_idx) in enumerate(skf.split(X, T), 1):
#     X_tr, X_te = X[tr_idx], X[te_idx]
#     T_tr, T_te = T[tr_idx], T[te_idx]
#     Y_tr, Y_te = Y[tr_idx], Y[te_idx]

#     # 1. 傾向スコアモデル P(T=1|X)
#     grid_ps = GridSearchCV(LogisticRegression(random_state=random_state), param_grid_ps, cv=3, scoring='roc_auc', n_jobs=-1)
#     grid_ps.fit(X_tr, T_tr)
#     df.loc[te_idx, 'ps_oof'] = grid_ps.predict_proba(X_te)[:, 1]

#     # 2. RORR用結果モデル E[Y|X]
#     grid_g = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_g.fit(X_tr, Y_tr)
#     df.loc[te_idx, 'g_hat_oof'] = grid_g.predict(X_te)

#     # 3. AIPW用結果モデル E[Y|T=0,X] と E[Y|T=1,X]
#     grid_m0 = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
#     df.loc[te_idx, 'm0_hat_oof'] = grid_m0.predict(X_te)
    
#     grid_m1 = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
#     df.loc[te_idx, 'm1_hat_oof'] = grid_m1.predict(X_te)
    
#     print(f"[Fold {fold:02d}/{n_splits}] サンプル外予測値を計算しました。")

# print("✅ 全てのサンプル外予測値の計算が完了しました。")

# plt.figure(figsize=(10, 6))
# sns.histplot(data=df, x='ps_oof', hue='t', bins=30, stat='density', common_norm=False, element='step')
# plt.title('Propensity Score Distribution (OOF)')
# plt.xlabel('Propensity Score')
# plt.ylabel('Density')
# plt.grid(True)
# plt.show()

# # --- 推定量を計算する関数群 ---
# def estimate_ipw_wls(df, t_name='t', y_name='y'):
#     ps_clipped = np.clip(df['ps_oof'], 0.05, 0.95)
#     weights = np.where(df[t_name] == 1, 1 / ps_clipped, 1 / (1 - ps_clipped))
#     X = sm.add_constant(df[t_name])
#     model = sm.WLS(df[y_name], X, weights=weights).fit()
#     return model.params.iloc[1], model.conf_int().iloc[1], model.pvalues.iloc[1]

# def estimate_rorr_standard(df, t_name='t', y_name='y'):
#     t_residual = df[t_name] - df['ps_oof']
#     y_residual = df[y_name] - df['g_hat_oof']
#     X = sm.add_constant(t_residual)
#     model = sm.OLS(y_residual, X).fit()
#     return model.params.iloc[1], model.conf_int().iloc[1], model.pvalues.iloc[1]
    
# def estimate_aipw_standard(df, t_name='t', y_name='y'):
#     T, Y, ps = df[t_name], df[y_name], np.clip(df['ps_oof'], 0.01, 0.99)
#     m0, m1 = df['m0_hat_oof'], df['m1_hat_oof']
#     psi1 = np.mean((T / ps) * (Y - m1) + m1)
#     psi0 = np.mean(((1 - T) / (1 - ps)) * (Y - m0) + m0)
#     ate = psi1 - psi0
    
#     if1 = (T / ps) * (Y - m1) + m1 - psi1
#     if0 = ((1 - T) / (1 - ps)) * (Y - m0) + m0 - psi0
#     influence_function = if1 - if0
#     se = np.sqrt(np.var(influence_function, ddof=1) / len(df))
    
#     ci = (ate - 1.96 * se, ate + 1.96 * se)
#     z_score = ate / se
#     p_value = 2 * (1 - norm.cdf(abs(z_score)))
#     return ate, ci, p_value

# # --- 推定の実行と結果の表示 ---
# ipw_effect, ipw_ci, ipw_p = estimate_ipw_wls(df)
# rorr_effect, rorr_ci, rorr_p = estimate_rorr_standard(df)
# aipw_effect, aipw_ci, aipw_p = estimate_aipw_standard(df)

# print("\n" + "="*50)
# print("           因果効果の推定結果サマリー")
# print("="*50)
    
# print(f"\n▶ IPW (WLS) 推定量")
# print(f"  ATE: {ipw_effect:.2f}")
# print(f"  95% CI: ({ipw_ci[0]:.2f}, {ipw_ci[1]:.2f})")
# print(f"  P-value: {ipw_p:.4g}")

# print(f"\n▶ RORR (標準) 推定量")
# print(f"  ATE: {rorr_effect:.2f}")
# print(f"  95% CI: ({rorr_ci[0]:.2f}, {rorr_ci[1]:.2f})")
# print(f"  P-value: {rorr_p:.4g}")
    
# print(f"\n▶ AIPW (標準/二重頑健) 推定量 ✨")
# print(f"  ATE: {aipw_effect:.2f}")
# print(f"  95% CI: ({aipw_ci[0]:.2f}, {aipw_ci[1]:.2f})")
# print(f"  P-value: {aipw_p:.4g}")

# print("="*50)