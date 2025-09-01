import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================================================================
# ユーザー自身のデータフレームに置き換えてください
# 例:
# df1 = pd.read_csv('treatment_group.csv')
# df2 = pd.read_csv('control_group.csv')
# ==============================================================================

# サンプルデータフレームを2つ作成
# df1を処置群、df2を対照群と仮定
np.random.seed(0)
df1 = pd.DataFrame({
    'ps': np.random.normal(0.4, 0.1, 1000) # 平均0.4の傾向スコアを持つ処置群
})
df2 = pd.DataFrame({
    'ps': np.random.normal(0.6, 0.1, 1000) # 平均0.6の傾向スコアを持つ対照群
})

# 傾向スコアは通常0から1の範囲のため、範囲外の値をクリップ
df1['ps'] = df1['ps'].clip(0, 1)
df2['ps'] = df2['ps'].clip(0, 1)


# === グラフの描画 ===

# グラフのサイズを設定
plt.figure(figsize=(10, 6))

# df1（処置群）の'ps'カラムの分布をプロット
sns.kdeplot(df1['ps'], label='DataFrame 1 (処置群)', fill=True, color='blue', alpha=0.5)

# df2（対照群）の'ps'カラムの分布をプロット
sns.kdeplot(df2['ps'], label='DataFrame 2 (対照群)', fill=True, color='orange', alpha=0.5)

# グラフのタイトルとラベルを設定
plt.title('傾向スコアの分布の比較', fontsize=16)
plt.xlabel('傾向スコア', fontsize=12)
plt.ylabel('密度', fontsize=12)

# 凡例を表示
plt.legend()

# グリッド線を表示
plt.grid(axis='y', linestyle='--', alpha=0.7)

# グラフを画像ファイルとして保存
plt.savefig('propensity_score_distribution.png')

# 実行完了メッセージ
print("グラフが 'propensity_score_distribution.png' として保存されました。")