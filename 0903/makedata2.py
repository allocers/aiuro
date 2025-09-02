import pandas as pd
import numpy as np

np.random.seed(42)  # 再現性のため

num_records = 1000

# 共通の分布をベースに生成
base_settlement = np.random.normal(80000, 20000, num_records*2)  # 平均8万, SD2万
base_points_use = np.random.normal(400, 100, num_records*2)
base_points_give = np.random.normal(250, 80, num_records*2)
base_card_use = np.random.normal(200000, 50000, num_records*2)
base_age = np.random.randint(20, 70, num_records*2)
base_contract_period = np.random.choice([1, 2, 3], num_records*2)
base_gender = np.random.choice([1, 2], num_records*2)  # 0=女性,1=男性

# コントロール群
control_data = {
    'COMNID': np.arange(1, num_records + 1),
    'DPAY_SETTLEMENT_AMOUNT_202406': base_settlement[:num_records] + np.random.normal(-5000, 5000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202407': base_settlement[:num_records] + np.random.normal(0, 10000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202408': base_settlement[:num_records] + np.random.normal(0, 12000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202409': base_settlement[:num_records] + np.random.normal(0, 15000, num_records),
    'DPOINTS_USE_202406': base_points_use[:num_records] + np.random.normal(-30, 20, num_records),
    'DPOINTS_USE_202407': base_points_use[:num_records] + np.random.normal(0, 30, num_records),
    'DPOINTS_USE_202408': base_points_use[:num_records] + np.random.normal(0, 30, num_records),
    'DPOINTS_USE_202409': base_points_use[:num_records] + np.random.normal(0, 30, num_records),
    'DPOINTS_GIVE_202406': base_points_give[:num_records] + np.random.normal(-20, 15, num_records),
    'DPOINTS_GIVE_202407': base_points_give[:num_records] + np.random.normal(0, 20, num_records),
    'DPOINTS_GIVE_202408': base_points_give[:num_records] + np.random.normal(0, 20, num_records),
    'DPOINTS_GIVE_202409': base_points_give[:num_records] + np.random.normal(0, 20, num_records),
    'DCARD_USE_AMOUNT_202406': base_card_use[:num_records] + np.random.normal(-20000, 10000, num_records),
    'CONTRACT': np.ones(num_records, dtype=int),
    'CONTRACT_PERIOD': base_contract_period[:num_records],
    'GENDER': base_gender[:num_records],
    'AGE': base_age[:num_records]
}
df_control = pd.DataFrame(control_data)

# トリートメント群（平均を少しシフト）
treatment_data = {
    'COMNID': np.arange(1, num_records + 1),
    'DPAY_SETTLEMENT_AMOUNT_202406': base_settlement[num_records:] + np.random.normal(5000, 5000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202407': base_settlement[num_records:] + np.random.normal(10000, 10000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202408': base_settlement[num_records:] + np.random.normal(12000, 12000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202409': base_settlement[num_records:] + np.random.normal(15000, 15000, num_records),
    'DPOINTS_USE_202406': base_points_use[num_records:] + np.random.normal(30, 20, num_records),
    'DPOINTS_USE_202407': base_points_use[num_records:] + np.random.normal(50, 30, num_records),
    'DPOINTS_USE_202408': base_points_use[num_records:] + np.random.normal(50, 30, num_records),
    'DPOINTS_USE_202409': base_points_use[num_records:] + np.random.normal(50, 30, num_records),
    'DPOINTS_GIVE_202406': base_points_give[num_records:] + np.random.normal(20, 15, num_records),
    'DPOINTS_GIVE_202407': base_points_give[num_records:] + np.random.normal(30, 20, num_records),
    'DPOINTS_GIVE_202408': base_points_give[num_records:] + np.random.normal(30, 20, num_records),
    'DPOINTS_GIVE_202409': base_points_give[num_records:] + np.random.normal(30, 20, num_records),
    'DCARD_USE_AMOUNT_202406': base_card_use[num_records:] + np.random.normal(20000, 10000, num_records),
    'CONTRACT': np.ones(num_records, dtype=int),
    'CONTRACT_PERIOD': base_contract_period[num_records:],
    'GENDER': base_gender[num_records:],
    'AGE': base_age[num_records:]
}
df_treatment = pd.DataFrame(treatment_data)

# 保存
df_control.to_csv('CPID5435_control2.csv', index=False)
df_treatment.to_csv('CPID5435_treatment2.csv', index=False)

print("分布が重なりつつ、群ごとに平均がずれたデータを生成しました。")
