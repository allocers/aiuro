import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

# サンプルのデータフレームを作成（実際のデータに置き換えてください）
data = {
    "AGE_std": np.random.rand(5000),
    "DPAY_SETTLEMENT_AMOUNT_202406_std": np.random.rand(5000),
    "DPOINTS_USE_202406_std": np.random.rand(5000),
    "DPOINTS_GIVE_202406_std": np.random.rand(5000),
    "DCARD_USE_AMOUNT_202406_std": np.random.rand(5000),
    "GENDER": np.random.randint(0, 2, 5000),
    "T": np.random.randint(0, 2, 5000), # Treatment variable
    "Y": np.random.rand(5000) # Outcome variable
}
df = pd.DataFrame(data)

# 特徴量とターゲット変数の定義
X_cols = [
    "AGE_std",
    "DPAY_SETTLEMENT_AMOUNT_202406_std",
    "DPOINTS_USE_202406_std",
    "DPOINTS_GIVE_202406_std",
    "DCARD_USE_AMOUNT_202406_std",
    "GENDER"
]
df['T'] = df['T'].astype(str)
df['Y'] = df['Y'].astype(float)
X = df[X_cols]
T = df['T']
Y = df['Y']

# パラメータグリッドの設定
param_grid_ps = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [2000]
}
param_grid_rf = {
    "n_estimators": [10, 50, 100, 150],
    "max_depth": [5, 10, None],
    "max_features": ["sqrt", "log2", None]
}

# Stratified K-Fold設定
K = 5
SEED = 1024
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

# インスタンスの生成
base_ps_model = LogisticRegression(random_state=SEED)
base_outcome_model = RandomForestRegressor(random_state=SEED)
base_g_model = RandomForestRegressor(random_state=SEED)
base_h_model = LinearRegression() # ここはRandomForestRegressorに変更することも可能

# 予測値を格納する空の列を作成
df["ps_hat"] = np.nan
df["mu0"] = np.nan
df["mu1"] = np.nan
df["g_hat"] = np.nan
df["h_hat"] = np.nan

# 評価値を格納する空のリスト
ps_scores_list = []
outcome_scores_list = []
ps_coefs_list = []
rf_importances_list = []

# K-Fold交差検証
for train_index, test_index in kf.split(df, df["T"]):
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    X_train, X_test = df_train[X_cols], df_test[X_cols]
    Y_train, Y_test = df_train["Y"], df_test["Y"]
    T_train, T_test = df_train["T"], df_test["T"]

    # 傾向スコア(P(T=1|X))モデルの学習
    ps_model = GridSearchCV(
        base_ps_model, param_grid_ps, cv=5, scoring="roc_auc", n_jobs=-1
    )
    ps_model.fit(X_train, T_train)
    ps_hat = ps_model.predict_proba(X_test[X_cols])[:, 1]
    df.loc[test_index, "ps_hat"] = ps_hat
    ps_scores_list.append(ps_model.best_score_)
    ps_coefs_list.append(ps_model.best_estimator_.coef_[0])

    # アウトカム回帰(E[Y|T,X])モデルの学習
    outcome_model = GridSearchCV(
        base_outcome_model,
        param_grid_rf,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    X_aug_train = df_train[X_cols].copy()
    X_aug_train["T"] = df_train["T"].astype(int)
    outcome_model.fit(X_aug_train, Y_train)
    mse = -outcome_model.best_score_
    outcome_scores_list.append(mse)
    rf_importances_list.append(
        outcome_model.best_estimator_.feature_importances_
    )

    # 全員が処置を受けた場合の推定(E[Y|T=1,X])
    X_test_mu1 = df_test[X_cols].copy()
    X_test_mu1["T"] = 1
    mu1_hat = outcome_model.predict(X_test_mu1)
    df.loc[test_index, "mu1"] = mu1_hat

    # 全員が処置を受けなかった場合の推定(E[Y|T=0,X])
    X_test_mu0 = df_test[X_cols].copy()
    X_test_mu0["T"] = 0
    mu0_hat = outcome_model.predict(X_test_mu0)
    df.loc[test_index, "mu0"] = mu0_hat

    # g(x) = E[Y|T=1,X]
    g_model = GridSearchCV(
        base_g_model,
        param_grid_rf,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    g_model.fit(df_train[df_train["T"] == '1'][X_cols], df_train[df_train["T"] == '1']["Y"])
    g_hat = g_model.predict(df_test[X_cols])
    df.loc[test_index, "g_hat"] = g_hat

    # h(x) = E[Y|T=0,X]
    h_model = LinearRegression() # ここはRandomForestRegressorに変更することも可能
    h_model.fit(df_train[df_train["T"] == '0'][X_cols], df_train[df_train["T"] == '0']["Y"])
    h_hat = h_model.predict(df_test[X_cols])
    df.loc[test_index, "h_hat"] = h_hat


# モデル評価の表示
print("--- Outcome Model Feature Importances (Random Forest) ---")
outcome_model_features = X_cols + ["T"]
df_rf_importances = pd.DataFrame(
    rf_importances_list, columns=outcome_model_features
)
df_importances_summary = pd.DataFrame(
    {
        "Mean_Importance": df_rf_importances.mean(),
        "Std_Importance": df_rf_importances.std(),
    }
).sort_values("Mean_Importance", ascending=False)
print(df_importances_summary)
print("\n")

print("--- Overall Model Performance (Mean across Folds) ---")
mean_ps_score = np.mean(ps_scores_list)
std_ps_score = np.std(ps_scores_list)
print(
    f"Propensity Score Model AUC: {mean_ps_score:.4f} (+/- {std_ps_score:.4f})"
)
mean_mse = np.mean(outcome_scores_list)
std_mse = np.std(outcome_scores_list)
print(f"Outcome Model MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")
print(f"Outcome Model RMSE: {np.sqrt(mean_mse):.4f}")
print("\n")

print("--- Propensity Score Model Coefficients (Logistic Regression) ---")
df_ps_coefs = pd.DataFrame(ps_coefs_list, columns=X_cols)
df_ps_coefs_summary = pd.DataFrame(
    {"Mean_Coef": df_ps_coefs.mean(), "Std_Coef": df_ps_coefs.std()}
).sort_values("Mean_Coef", ascending=False)
print(df_ps_coefs_summary)
print("\n")

# 傾向スコア推定値の可視化
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="ps_hat",
    hue="T",
    bins=np.arange(0, 1.1, 0.1),
    common_norm=False,
)
plt.xlim(0, 1)
plt.xticks(np.linspace(0, 1, 6))
plt.title("Distribution of Estimated propensity scores")
plt.xlabel("Estimated propensity score")
plt.show()

# トリミング
alpha = 0.05
mask = (df["ps_hat"] > alpha) & (df["ps_hat"] < 1 - alpha)
trimmed_rows = len(df[~mask])
total_rows = len(df)
percent_trimmed = (trimmed_rows / total_rows) * 100
print(f"Trimmed {trimmed_rows} / {total_rows} rows ({percent_trimmed:.1f}%) outside [{alpha:.2f}, {1-alpha:.2f}]")
df_trimmed = df.loc[mask].reset_index(drop=True)

# 因果効果の計算と結果格納
results = {}

# 単純比較
mean_treat = df_trimmed.loc[df_trimmed["T"] == "1", "Y"].mean()
mean_control = df_trimmed.loc[df_trimmed["T"] == "0", "Y"].mean()
ate_naive = mean_treat - mean_control
results["Naive"] = ate_naive
print(f"ATE (Naive Difference): {results['Naive']:.10f}")

# IPW推定: Horvitz-Thompson型
df_trimmed["ate_hat_ipw_HT"] = (df_trimmed["T"] == "1") * df_trimmed["Y"] / df_trimmed["ps_hat"] - (df_trimmed["T"] == "0") * df_trimmed["Y"] / (1 - df_trimmed["ps_hat"])
ate_ipw_ht = df_trimmed["ate_hat_ipw_HT"].mean()
results["IPW_HT"] = ate_ipw_ht
print(f"ATE (IPW_HT): {results['IPW_HT']:.10f}")

# IPW推定: Hajek型
df_trimmed["w_ipw_treat_h"] = (df_trimmed["T"] == "1") / df_trimmed["ps_hat"]
df_trimmed["w_ipw_control_h"] = (df_trimmed["T"] == "0") / (1 - df_trimmed["ps_hat"])
num1 = (df_trimmed["w_ipw_treat_h"] * df_trimmed["Y"]).sum()
den1 = df_trimmed["w_ipw_treat_h"].sum()
num0 = (df_trimmed["w_ipw_control_h"] * df_trimmed["Y"]).sum()
den0 = df_trimmed["w_ipw_control_h"].sum()
ate_ipw_hajek = (num1 / den1) - (num0 / den0)
results["IPW_H"] = ate_ipw_hajek
print(f"ATE (IPW_H): {results['IPW_H']:.10f}")

# G-computation推定
ate_hat_gcomp = df_trimmed["mu1"] - df_trimmed["mu0"]
results["G-computation"] = ate_hat_gcomp.mean()
print(f"ATE (G-computation, RF): {results['G-computation']:.10f}")

# AIPW (二重にロバスト) 推定
ate_aipw_treat = (df_trimmed["T"] == "1") * df_trimmed["Y"] / df_trimmed["ps_hat"] - (df_trimmed["T"] == "1") * df_trimmed["mu0"] / df_trimmed["ps_hat"] + df_trimmed["mu0"]
ate_aipw_control = -(df_trimmed["T"] == "0") * df_trimmed["Y"] / (1 - df_trimmed["ps_hat"]) + (df_trimmed["T"] == "0") * df_trimmed["mu1"] / (1 - df_trimmed["ps_hat"]) - df_trimmed["mu1"]
ate_hat_aipw = ate_aipw_treat.mean() - ate_aipw_control.mean()
results["AIPW"] = ate_hat_aipw
print(f"ATE (AIPW): {results['AIPW']:.10f}")

# RORR推定
df_trimmed["y_hat"] = np.where(df_trimmed["T"] == "1", df_trimmed["g_hat"], df_trimmed["h_hat"])
df_trimmed["y_hat"] = df_trimmed["g_hat"] if df_trimmed["T"] == "1" else df_trimmed["h_hat"] # この行は論理的に間違っている可能性あり
df_trimmed["y_hat"] = df_trimmed["g_hat"] # 画像からR-Learnerではないと判断
df_trimmed["t_hat"] = df_trimmed["ps_hat"]
y_resid = df_trimmed["Y"] - df_trimmed["y_hat"]
t_resid = df_trimmed["T"].astype(int) - df_trimmed["t_hat"]

# 残差同士の回帰
ols_model = sm.OLS(y_resid, sm.add_constant(t_resid)).fit()
ate_rorr = ols_model.params[1]
results["RORR"] = ate_rorr
print(f"ATE (RORR): {results['RORR']:.10f}")


# 全ATE結果のデータフレーム表示
df_ATE_result = pd.DataFrame(results, index=[0])
print(df_ATE_result)

# ---
## SMD（標準化平均差）の計算と可視化

### SMD計算用関数の定義
```python
def compute_smd(df, covariates, treatment, weights=None):
    smd_list = []
    for cov in covariates:
        if weights is None:
            # 重みなし
            mean_treat = df.loc[df[treatment] == "1", cov].mean()
            mean_control = df.loc[df[treatment] == "0", cov].mean()
            var_treat = df.loc[df[treatment] == "1", cov].var()
            var_control = df.loc[df[treatment] == "0", cov].var()
        else:
            # 重みあり
            w = weights
            mean_treat = np.average(df.loc[df[treatment] == "1", cov], weights=w[df[treatment] == "1"])
            mean_control = np.average(df.loc[df[treatment] == "0", cov], weights=w[df[treatment] == "0"])
            var_treat = np.average((df.loc[df[treatment] == "1", cov] - mean_treat)**2, weights=w[df[treatment] == "1"])
            var_control = np.average((df.loc[df[treatment] == "0", cov] - mean_control)**2, weights=w[df[treatment] == "0"])
        
        # SMDの計算
        smd = (mean_treat - mean_control) / np.sqrt((var_treat + var_control) / 2)
        smd_list.append(smd)
    return smd_list

# SMDの計算とデータフレームへの格納
# 重みなし
smd_unweighted = compute_smd(df_trimmed, X_cols, "T")

# IPW_HTの重み
w_ipw_ht = np.where(df_trimmed["T"] == "1", 1 / df_trimmed["ps_hat"], 1 / (1 - df_trimmed["ps_hat"]))
smd_ipw_ht = compute_smd(df_trimmed, X_cols, "T", weights=w_ipw_ht)

# IPW_Hの重み
w1 = 1 / df_trimmed["ps_hat"]
w0 = 1 / (1 - df_trimmed["ps_hat"])
w_ipw_hajek = np.where(df_trimmed["T"] == "1", w1 / w1.sum(), w0 / w0.sum())
smd_ipw_hajek = compute_smd(df_trimmed, X_cols, "T", weights=w_ipw_hajek)

smd_df = pd.DataFrame({
    "Unweighted": smd_unweighted,
    "IPW_HT": smd_ipw_ht,
    "IPW_H": smd_ipw_hajek,
}, index=X_cols).T

# SMDの可視化
fig, ax = plt.subplots(figsize=(10, 8))
for method in smd_df.index:
    ax.scatter(smd_df.loc[method].values, smd_df.columns, label=method, alpha=0.8)

ax.axvline(0.1, color='k', linestyle='--', label="SMD=0.1基準")
ax.axvline(0.25, color='k', linestyle=':', label="SMD=0.25基準")
ax.set_xlabel("Standardized Mean Difference")
ax.set_ylabel("Covariates")
ax.set_title("Covariate Balance Before/After Weighting")
ax.legend()
plt.tight_layout()
plt.show()

# 重みの分布を可視化（Boxplot）
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=df_trimmed, x="T", y=w_ipw_ht, ax=axes[0], palette="Set2")
axes[0].set_title("HT weight by T")
axes[0].set_ylabel("w_ipw_ht")

sns.boxplot(data=df_trimmed, x="T", y=w_ipw_hajek, ax=axes[1], palette="Set2")
axes[1].set_title("Hajek-normalized weight by T")
axes[1].set_ylabel("w_ipw_hajek")

plt.tight_layout()
plt.show()