import numpy as np
import pandas as pd

# --- 設定 ---
NUM_COVARIATES = 10  # 使用する共変量の数
BETA_X = np.random.uniform(-0.2, 0.2, size=NUM_COVARIATES) # アウトカムモデルの係数
GAMMA_X = np.random.uniform(-0.3, 0.3, size=NUM_COVARIATES) # 処置モデルの係数


def g_x(x_df, beta_x=BETA_X):
    """Y = f(t) + g(x) + e における g(x)"""
    return x_df.values @ beta_x

def h_x(x_df, gamma_x=GAMMA_X):
    """T = h(x) + u における h(x)"""
    return np.exp(x_df.values @ gamma_x)

def f_t(t_array):
    """Y = f(t) + g(x) + e における f(t)"""
    return np.log(t_array + 1)

def generate_x(sample_size, num_covariates, num_strata, seed=None):
    """複数の共変量を生成"""
    rng = np.random.default_rng(seed)
    covariates = {
        f'x{i+1}': rng.integers(1, num_strata + 1, size=sample_size)
        for i in range(num_covariates)
    }
    return pd.DataFrame(covariates)

def generate_t(x_df, seed=None):
    """処置変数Tを生成"""
    rng = np.random.default_rng(seed)
    lambda_vals = h_x(x_df)
    return rng.poisson(lam=lambda_vals)

def generate_y(t_array, x_df, seed=None):
    """アウトカム変数Yを生成"""
    rng = np.random.default_rng(seed)
    e = rng.normal(loc=0.0, scale=1.0, size=len(t_array))
    return f_t(t_array) + g_x(x_df) + e

def append_derivative(df):
    """log(t + 1)の導関数を追加"""
    df = df.copy()
    df['derivative'] = 1 / (df['t'] + 1)
    return df

def append_incremental(df):
    """tを1増加させたときの差分効果を追加"""
    df = df.copy()
    df['incremental'] = np.log(df['t'] + 2) - np.log(df['t'] + 1)
    return df

def append_t_star(df):
    """有効処置点t_starを追加"""
    df = df.copy()
    t = df['t'].to_numpy()
    covariate_cols = [col for col in df.columns if col.startswith('x')]
    x_summary = h_x(df[covariate_cols])

    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.log((t + 1) / (x_summary + 1))
        t_star = np.where(
            np.isclose(t, x_summary) | np.isclose(log_ratio, 0),
            t,
            (t - x_summary) / log_ratio - 1
        )
    df['t_star'] = np.nan_to_num(t_star, nan=t.astype(float))
    return df

def append_conditional_variance_weight(df):
    """処置残差の分散に基づいた重みを追加"""
    df = df.copy()
    t = df['t'].to_numpy()
    covariate_cols = [col for col in df.columns if col.startswith('x')]
    x_summary = h_x(df[covariate_cols])
    
    residuals = t - x_summary
    mean_sq_resid = np.mean(residuals ** 2)
    if mean_sq_resid == 0:
        df['weight'] = 1.0
    else:
        df['weight'] = residuals ** 2 / mean_sq_resid
    return df

def simulate_dataset(sample_size, num_covariates, num_strata, seed=None):
    """データセット全体をシミュレーション"""
    x_df = generate_x(sample_size, num_covariates, num_strata, seed)
    t = generate_t(x_df, seed)
    y = generate_y(t, x_df, seed)

    data = pd.concat([pd.DataFrame({'y': y, 't': t}), x_df], axis=1)

    return (
        data
        .pipe(append_derivative)
        .pipe(append_incremental)
        .pipe(append_t_star)
        .pipe(append_conditional_variance_weight)
        .assign(plm_plim=lambda df: df.weight * 1 / (df.t_star + 1))
    )