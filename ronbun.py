import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 1. ダミーデータの生成 ---
np.random.seed(42)
n_samples = 2000

# 共変量: ユーザーの「介入前訪問回数」を模倣
# 介入群と対照群で意図的に分布に差を設ける
pre_treatment_visits = np.random.normal(loc=5, scale=2, size=n_samples) # 全体の分布

# 介入 (Treatment): バイアスを導入
# pre_treatment_visitsが多いユーザーほど介入を受ける確率が高いとする
treatment_prob = 1 / (1 + np.exp(-(pre_treatment_visits - 5) * 0.5))
treatment = np.random.binomial(1, treatment_prob) # 0: 対照群, 1: 介入群

# 別の特徴量A (縦軸のビンに対応させる)
# これは共変量とは独立しているが、バランスを見るためのグループ分けに使う
feature_a_usage = np.random.randint(1, 6, size=n_samples) # 1から5の整数ビン

# データフレームにまとめる
df = pd.DataFrame({
    'pre_treatment_visits': pre_treatment_visits,
    'treatment': treatment,
    'feature_a_usage_bin': feature_a_usage
})

# --- 2. 傾向スコアの推定 ---
# 傾向スコアモデルの学習 (共変量から介入確率を予測)
# ここでは pre_treatment_visits を唯一の共変量とする
propensity_model = LogisticRegression(solver='lbfgs', C=0.1)
propensity_model.fit(df[['pre_treatment_visits']], df['treatment'])

# 傾向スコア (e(X) = P(T=1|X)) を取得
propensity_scores = propensity_model.predict_proba(df[['pre_treatment_visits']])[:, 1]
propensity_scores = np.clip(propensity_scores, 0.01, 0.99) # 0や1に張り付くのを防ぐ

# --- 3. IPWの計算 ---
# IPW = T/e(X) + (1-T)/(1-e(X))
df['ipw'] = (df['treatment'] / propensity_scores) + ((1 - df['treatment']) / (1 - propensity_scores))

# --- 4. 標準化平均差 (SMD) の計算関数 ---
def calculate_smd(data, variable, treatment_col, weight_col=None):
    """
    指定された変数における介入群と対照群の標準化平均差を計算する。
    重み付けが指定された場合は、重み付き平均差を計算する。
    """
    if weight_col:
        # 重み付き平均と標準偏差
        t_mean = np.average(data.loc[data[treatment_col] == 1, variable], weights=data.loc[data[treatment_col] == 1, weight_col])
        c_mean = np.average(data.loc[data[treatment_col] == 0, variable], weights=data.loc[data[treatment_col] == 0, weight_col])
        
        # 標準化のためのプールされた標準偏差 (通常、IPW後のSMDでは、
        # 未調整時のコントロール群の標準偏差を使用することが多いが、
        # ここでは単純化のため両群の重み付き標準偏差のプールを使用)
        t_var = np.average((data.loc[data[treatment_col] == 1, variable] - t_mean)**2, weights=data.loc[data[treatment_col] == 1, weight_col])
        c_var = np.average((data.loc[data[treatment_col] == 0, variable] - c_mean)**2, weights=data.loc[data[treatment_col] == 0, weight_col])
        
        pooled_std = np.sqrt((t_var + c_var) / 2) # または対照群の標準偏差を使うことも
        # pooled_std = np.std(data.loc[data[treatment_col] == 0, variable]) # 対照群の標準偏差
    else:
        # 重みなし平均と標準偏差
        t_mean = data.loc[data[treatment_col] == 1, variable].mean()
        c_mean = data.loc[data[treatment_col] == 0, variable].mean()
        pooled_std = np.sqrt((data.loc[data[treatment_col] == 1, variable].var() + \
                               data.loc[data[treatment_col] == 0, variable].var()) / 2)
        # pooled_std = data.loc[data[treatment_col] == 0, variable].std() # 対照群の標準偏差

    if pooled_std == 0: return 0 # ゼロ除算回避
    return (t_mean - c_mean) / pooled_std


# 各feature_a_usage_binごとにSMDを計算
smd_results = []
for bin_val in sorted(df['feature_a_usage_bin'].unique()):
    subset = df[df['feature_a_usage_bin'] == bin_val].copy()
    
    # Before IPW
    smd_before = calculate_smd(subset, 'pre_treatment_visits', 'treatment')
    smd_results.append({'Bin': bin_val, 'Type': 'Before IPW', 'SMD': smd_before})
    
    # After IPW
    smd_after = calculate_smd(subset, 'pre_treatment_visits', 'treatment', weight_col='ipw')
    smd_results.append({'Bin': bin_val, 'Type': 'After IPW', 'SMD': smd_after})

smd_df = pd.DataFrame(smd_results)

# --- 5. プロット ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(data=smd_df, x='SMD', y='Bin', hue='Type', ax=ax, s=150, zorder=2,
                palette={'Before IPW': 'tab:orange', 'After IPW': 'tab:blue'})

# 基準線 (0.0) と許容範囲 (-0.1, 0.1) の描画
ax.axvline(x=0, color='red', linestyle='-', linewidth=0.8, zorder=1)
ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=0.8, zorder=1)
ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=0.8, zorder=1)

ax.set_title('Balance in Pre-Treatment Outcomes After Weighting', fontsize=16, pad=20)
ax.set_xlabel('Standardized Difference in Mean Pre-Treatment Visits', fontsize=12)
ax.set_ylabel('Feature A Usage Bins', fontsize=12)
ax.set_yticks(sorted(df['feature_a_usage_bin'].unique())) # 縦軸のラベルをビンに対応させる
ax.set_xlim(-0.5, 2.0) # 論文の図に合わせて横軸の範囲を調整
ax.legend(title='Weighting Status', fontsize=11, loc='upper right')

plt.tight_layout()
plt.show()

print("\n--- Standardized Mean Differences ---")
print(smd_df)