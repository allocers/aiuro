import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
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


print(df.info())
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
plt.show()

# # Standardize the specified columns
scaler = StandardScaler()
cols_to_standardize = ['DPAY_SETTLEMENT_AMOUNT_202406', 'DPOINTS_USE_202406','DPOINTS_USE_202407','DPOINTS_USE_202408','DPOINTS_USE_202409','DPOINTS_GIVE_202406', 'DPOINTS_GIVE_202407', 'DPOINTS_GIVE_202408', 'DPOINTS_GIVE_202409', 'DCARD_USE_AMOUNT_202406']
df[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])
# =========================
# 10-fold CVでPS推定 + IPW集計一式
# =========================

# --- ① ロジスティック回帰の10fold CVでOOF傾向スコアを作る ---
def crossval_propensity_scores(df, features, t_name='t', n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    ps_oof = np.zeros(len(df), dtype=float)
    aucs = []

    X = df[features].values
    y = df[t_name].values

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X[tr_idx], y[tr_idx])
        ps_fold = model.predict_proba(X[te_idx])[:, 1]
        ps_oof[te_idx] = ps_fold

        # foldのAUC（テスト側で測定）
        auc = roc_auc_score(y[te_idx], ps_fold)
        aucs.append(auc)
        print(f"[Fold {fold:02d}] AUC = {auc:.4f}")

    print(f"AUC (mean ± sd) = {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    return ps_oof, aucs

# --- ② クリッピング関数 ---
def clip_ps(ps, thr=0.05):
    return np.clip(ps, thr, 1.0 - thr)

# --- ③ IPW（プラグイン推定） ---
def IPW(df, y_name, t_name, ps_name, thr=0.05):
    # クリッピング
    ps = df[ps_name].astype(float).values
    ps = clip_ps(ps, thr=thr)

    t = df[t_name].astype(int).values
    y = df[y_name].astype(float).values

    # 重み
    w1  = t / ps
    w0  = (1 - t) / (1 - ps)
    w01 = ((1 - t) * ps) / (1 - ps)
    w10 = (t * (1 - ps)) / ps

    # 期待値
    E1  = np.sum(y * w1) / np.sum(w1)     # E[Y1]
    E0  = np.sum(y * w0) / np.sum(w0)     # E[Y0]
    E01 = np.sum(y * w01) / np.sum(w01)   # E[Y0 | Z=1]
    E10 = np.sum(y * w10) / np.sum(w10)   # E[Y1 | Z=0]
    E11 = np.mean(y[t == 1])              # E[Y1 | Z=1]
    E00 = np.mean(y[t == 0])              # E[Y0 | Z=0]

    SIMPLE = E11 - E00
    ATE    = E1 - E0
    ATT    = E11 - E01
    ATU    = E10 - E00
    return {
        "SIMPLE": SIMPLE,
        "ATE": ATE,
        "ATT": ATT,
        "ATU": ATU,
        "components": dict(E1=E1, E0=E0, E01=E01, E10=E10, E11=E11, E00=E00)
    }

# --- ④ WLSで効果・p値・95%CI（ATE/ATT/ATU） ---
import statsmodels.api as sm

def wls_effect(df, y_name, t_name, ps_name, thr=0.05, kind='ATE'):
    """
    kind in {'ATE','ATT','ATU'}
    ATE: weights = 1/ps for t=1, 1/(1-ps) for t=0
    ATT: weights = 1 for t=1, ps/(1-ps) for t=0
    ATU: weights = (1-ps)/ps for t=1, 1 for t=0
    """
    y = df[y_name].astype(float).values
    t = df[t_name].astype(int).values
    ps = clip_ps(df[ps_name].astype(float).values, thr=thr)

    if kind == 'ATE':
        w = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    elif kind == 'ATT':
        w = np.where(t == 1, 1.0, ps / (1.0 - ps))
    elif kind == 'ATU':
        w = np.where(t == 1, (1.0 - ps) / ps, 1.0)
    else:
        raise ValueError("kind must be one of {'ATE','ATT','ATU'}")

    X = sm.add_constant(t)  # 切片 + 処置変数
    model = sm.WLS(y, X, weights=w)
    res = model.fit()

    beta = res.params[1]              # tの係数
    pval = res.pvalues[1]
    ci_l, ci_u = res.conf_int(alpha=0.05)[1]  # 95%CI
    return {"effect": beta, "pvalue": pval, "ci": (ci_l, ci_u), "summary": res}

# --- ⑤ 単純比較（差の平均）の95%CI ---
def simple_diff_ci(df, y_name='y', t_name='t', alpha=0.05):
    y1 = df.loc[df[t_name] == 1, y_name].values
    y0 = df.loc[df[t_name] == 0, y_name].values
    diff = np.mean(y1) - np.mean(y0)
    se = np.sqrt(np.var(y1, ddof=1)/len(y1) + np.var(y0, ddof=1)/len(y0))
    z = norm.ppf(1 - alpha/2)
    ci = (diff - z*se, diff + z*se)
    return {"effect": diff, "ci": ci, "se": se}

# --- ⑥ 実行：PS推定→図作成→推定量の出力 ---
# ロジスティック回帰で10-fold CV（OOFのpsを作成）
df['ps_oof'], aucs = crossval_propensity_scores(df, covariate_cols, t_name='t', n_splits=10, random_state=42)

# クリッピング閾値（必要に応じて変更）
thr = 0.05

# 傾向スコア分布（クリッピング前）
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='ps_oof', hue='t', bins=30, stat='density', element='step', common_norm=False)
plt.title('Propensity Score Distribution (OOF, unclipped)')
plt.xlabel('Propensity score')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('ps_distribution_unclipped.png', dpi=200)
plt.close()
print("傾向スコア分布（unclipped）を 'ps_distribution_unclipped.png' に保存しました。")

# 傾向スコア分布（クリッピング後）
df['ps_oof_clip'] = clip_ps(df['ps_oof'].values, thr=thr)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='ps_oof_clip', hue='t', bins=30, stat='density', element='step', common_norm=False)
plt.title(f'Propensity Score Distribution (OOF, clipped at {thr})')
plt.xlabel('Propensity score (clipped)')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('ps_distribution_clipped.png', dpi=200)
plt.close()
print("傾向スコア分布（clipped）を 'ps_distribution_clipped.png' に保存しました。")

# --- 単純比較 ---
simple = simple_diff_ci(df, y_name='y', t_name='t')
print(f"[単純比較] effect = {simple['effect']:.4f}, 95% CI = ({simple['ci'][0]:.4f}, {simple['ci'][1]:.4f})")

# --- IPW（プラグイン推定）: OOF ps + クリッピングでATE/ATT/ATU ---
ipw_est = IPW(df, y_name='y', t_name='t', ps_name='ps_oof', thr=thr)
print("[IPW plugin] ATE={:.4f}, ATT={:.4f}, ATU={:.4f}, SIMPLE={:.4f}".format(
    ipw_est["ATE"], ipw_est["ATT"], ipw_est["ATU"], ipw_est["SIMPLE"]
))

# --- IPW（WLS）: 効果・p値・95%CI ---
res_ate = wls_effect(df, y_name='y', t_name='t', ps_name='ps_oof', thr=thr, kind='ATE')
print("[WLS ATE] effect={:.4f}, p={:.4g}, 95% CI=({:.4f}, {:.4f})".format(
    res_ate["effect"], res_ate["pvalue"], res_ate["ci"][0], res_ate["ci"][1]
))

res_att = wls_effect(df, y_name='y', t_name='t', ps_name='ps_oof', thr=thr, kind='ATT')
print("[WLS ATT] effect={:.4f}, p={:.4g}, 95% CI=({:.4f}, {:.4f})".format(
    res_att["effect"], res_att["pvalue"], res_att["ci"][0], res_att["ci"][1]
))

res_atu = wls_effect(df, y_name='y', t_name='t', ps_name='ps_oof', thr=thr, kind='ATU')
print("[WLS ATU] effect={:.4f}, p={:.4g}, 95% CI=({:.4f}, {:.4f})".format(
    res_atu["effect"], res_atu["pvalue"], res_atu["ci"][0], res_atu["ci"][1]
))

# --- ⑦ foldごとのIPW ATEを見て、ばらつき（95%区間）を確認 ---
#    ここでは各foldの“テスト部分”だけでIPWを計算し、その分布を見る
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X = df[covariate_cols].values
y_t = df['t'].values
ps = df['ps_oof'].values

ate_folds = []
for tr_idx, te_idx in skf.split(X, y_t):
    sub = df.iloc[te_idx].copy()
    # そのfoldのテスト側のpsはOOF予測になっているのでそのまま使用
    est = IPW(sub, y_name='y', t_name='t', ps_name='ps_oof', thr=thr)
    ate_folds.append(est['ATE'])

ate_folds = np.array(ate_folds)
ate_mean = np.mean(ate_folds)
ci_l, ci_u = np.quantile(ate_folds, [0.025, 0.975])  # 分位で95%範囲を見る
print("[fold別ATE] mean={:.4f}, 95% range=({:.4f}, {:.4f})".format(ate_mean, ci_l, ci_u))



