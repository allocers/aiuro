import pandas as pd
import numpy as np

# Controlデータ生成
num_records = 1000
control_data = {
    'COMNID': np.arange(1, num_records + 1),
    'DPAY_SETTLEMENT_AMOUNT_202406': np.random.randint(0, 100000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202407': np.random.randint(0, 100000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202408': np.random.randint(0, 100000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202409': np.random.randint(50000, 150000, num_records),
    'DPOINTS_USE_202406': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202407': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202408': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202409': np.random.randint(100, 800, num_records),
    'DPOINTS_GIVE_202406': np.random.randint(0, 500, num_records),
    'DPOINTS_GIVE_202407': np.random.randint(0, 500, num_records),
    'DPOINTS_GIVE_202408': np.random.randint(0, 500, num_records),
    'DPOINTS_GIVE_202409': np.random.randint(0, 500, num_records),
    'DCARD_USE_AMOUNT_202406': np.random.randint(0, 300000, num_records),
    'CONTRACT': np.ones(num_records, dtype=int), # すべて1に設定
    'CONTRACT_PERIOD': np.random.choice([1, 2, 3], num_records),
    'GENDER': np.random.choice([1, 2], num_records),
    'AGE': np.random.randint(20, 70, num_records)
}

df_control = pd.DataFrame(control_data)
df_control.to_csv('CPID5435_control.csv', index=False)

# Treatmentデータ生成（別の内容）
treatment_data = {
    'COMNID': np.arange(1, num_records + 1),
    'DPAY_SETTLEMENT_AMOUNT_202406': np.random.randint(50000, 150000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202407': np.random.randint(50000, 150000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202408': np.random.randint(50000, 150000, num_records),
    'DPAY_SETTLEMENT_AMOUNT_202409': np.random.randint(50000, 150000, num_records),
    'DPOINTS_USE_202406': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202407': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202408': np.random.randint(100, 800, num_records),
    'DPOINTS_USE_202409': np.random.randint(100, 800, num_records),
    'DPOINTS_GIVE_202406': np.random.randint(100, 800, num_records),
    'DPOINTS_GIVE_202407': np.random.randint(100, 800, num_records),
    'DPOINTS_GIVE_202408': np.random.randint(100, 800, num_records),
    'DPOINTS_GIVE_202409': np.random.randint(100, 800, num_records),
    'DCARD_USE_AMOUNT_202406': np.random.randint(100000, 500000, num_records),
    'CONTRACT': np.ones(num_records, dtype=int), # すべて1に設定
    'CONTRACT_PERIOD': np.random.choice([1, 2, 3], num_records),
    'GENDER': np.random.choice([1, 2], num_records),
    'AGE': np.random.randint(20, 70, num_records)
}

df_treatment = pd.DataFrame(treatment_data)
df_treatment.to_csv('CPID5435_treatment.csv', index=False)

print('CPID5435_control.csv と CPID5435_treatment.csv を生成しました。')