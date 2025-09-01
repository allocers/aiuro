#共変量１つの時はget_rorr_estimatesとestimate_aipw_matrixを修正する

# Imports
import numpy as np
import pandas as pd
from scipy.stats import poisson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# --- Functions for Causal Inference Estimators
def compute_ocd(num_strata):
    """
    解析的なOCD (平均因果効果) を計算
    """
    strata = np.arange(1, num_strata + 1)
    return np.mean(g * (1 + np.exp(-strata)) / strata)

##共変量１つの時
# def get_rorr_estimates(data, g_model, h_model):
#     """
#     Residual-on-Residual Regression (RORR) with sklearn-style models.
    
#     Parameters:
#     - data: DataFrame with columns 'y', 't', 'x'
#     - g_model: sklearn model trained to predict Y from X (E[Y|X])
#     - h_model: sklearn model trained to predict T from X (E[T|X])
    
#     Returns:
#     - (coef, se, [lwr, upr]): coefficient, std.error, and 95% CI
#     """
#     X = data[['x']].to_numpy()
    
#     # Predict residuals
#     g_hat = g_model.predict(X)
#     h_hat = h_model.predict(X)
    
#     # Calculate residuals
#     y_hat = data['y'].to_numpy() - g_hat
#     t_hat = data['t'].to_numpy() - h_hat
    
#     # Residual-on-residual regression
#     fit = OLS(y_hat, add_constant(t_hat)).fit()
    
#     # Get coefficients, SE, and CI
#     if hasattr(fit.params, "iloc"):
#         # For numpy/Series, access by name
#         coef = fit.params.iloc[1]
#         se = fit.bse.iloc[1]
#         lwr, upr = fit.conf_int().iloc[1].values
#     else:
#         # For other index types, access by position
#         coef = fit.params[1]
#         se = fit.bse[1]
#         lwr, upr = fit.conf_int()[1]
        
#     return coef, se, [lwr, upr]


def get_rorr_estimates(data, g_model, h_model):
    """Residual-on-Residual Regression (RORR) with sklearn-style models."""
    covariate_cols = [col for col in data.columns if col.startswith('x')]
    X = data[covariate_cols]
    
    g_hat = g_model.predict(X)
    h_hat = h_model.predict(X)
    
    y_hat = data['y'].to_numpy() - g_hat
    t_hat = data['t'].to_numpy() - h_hat
    
    fit = OLS(y_hat, add_constant(t_hat)).fit()
    
    coef = fit.params[1]
    se = fit.bse[1]
    # ★★★ FIX: Use NumPy indexing instead of .iloc ★★★
    lwr, upr = fit.conf_int()[1]
        
    return coef, se, [lwr, upr]

def make_p_x(model):
    """
    Makes a function for the propensity score model.
    model: sklearn-style model (must implement .predict_proba)
    Returns a function that applies the model.
    """
    def p_x(x_array):
        # ensure x_array is 2D for sklearn models
        x_array = np.array(x_array).reshape(-1, 1)
        return model.predict_proba(x_array)
    return p_x


def make_m_tx(model):
    """
    ラップ関数: 与えられた x_array, t_array を結合して model.predict に渡す
    model: sklearn-style model (must implement .predict)
    Returns a function that applies the model.
    """
    def m_tx(x_array, t_array):
        x_array = np.array(x_array).reshape(-1, 1)
        t_array = np.array(t_array).reshape(-1, 1)
        input_aug = np.hstack([x_array, t_array])
        return model.predict(input_aug)
    return m_tx

##共変量１つの時
# def estimate_aipw_matrix(df, outcome_model, pscore_model, t_max=None):
#     """
#     Computes a wide-form AIPW influence matrix.
    
#     Parameters:
#     - df: DataFrame with columns 'y', 't', 'x' (use test data)
#     - outcome_model: sklearn-style model trained on (x,t) -> y
#     - pscore_model: sklearn-style model trained on x -> t
    
#     Returns:
#     - AIPW matrix as a DataFrame:
#       - index matches df.index
#       - columns are influence values for t = 0, 1, ..., max(t)
#     """
#     y = df['y'].to_numpy()
#     t_obs = df['t'].to_numpy()
#     x = df['x'].to_numpy()
#     idx = df.index
    
#     ### 中でラップする ###
#     m_model = make_m_tx(outcome_model) # outcome model wrapper
#     p_model = make_p_x(pscore_model) # propensity score wrapper
    
#     if t_max is None:
#         t_max = np.max(t_obs)
    
#     t_vals = np.arange(t_max + 1)
    
#     # --- Shape (n, len(t_vals))
#     m_vals = np.column_stack([m_model(x, np.full_like(x, t)) for t in t_vals])
    
#     # --- Shape (n, len(t_vals))
#     pscore_raw = p_model(x)
#     n, _ = pscore_raw.shape
#     pscore = np.zeros((n, len(t_vals)))
#     for i, t in enumerate(pscore_model.classes_):
#         if t < len(t_vals):
#             pscore[:,t] = pscore_raw[:, i]

#     # # pscore_model.classes_ exists if pscore is from a classifier with classes
#     # if pscore_model.classes_ is not None:
#     #     if pscore_raw.shape[1] != len(pscore_model.classes_):
#     #         raise ValueError("pscore_model must return probabilities for each t in t_vals.")
        
#     #     # Populate pscore matrix based on classes
#     #     for i, t in enumerate(pscore_model.classes_):
#     #         if t in t_vals:
#     #             pscore[:, t] = pscore_raw[:, i]

#     # Sanity check
#     if np.any(pscore.sum(axis=1) == 0):
#         raise ValueError("Some rows have zero probability across all classes!")
    
    
#     indicator = (t_obs[:,None]==t_vals[None, :]).astype(float)
#     # AIPW formula
#     influence = indicator / np.clip(pscore, 1e-12, None) * (y[:, None] - m_vals) + m_vals
    
#     # influence = np.zeros_like(pscore)
#     # for i, t in enumerate(t_vals):
#     #     indicator = (t_obs == t).astype(float)
#     #     influence[:, i] = indicator / np.clip(pscore[:, i], a_min=1e-12, a_max=None) * (y - m_vals[:, i]) + m_vals[:, i]
    
#     # DataFrame conversion
#     influence_df = pd.DataFrame(influence, index=idx, columns=[f"t_{t}" for t in t_vals])
    
#     return influence_df

def estimate_aipw_matrix(df, outcome_model, pscore_model, t_max=None):
    """Computes a wide-form AIPW influence matrix."""
    y = df['y'].to_numpy()
    t_obs = df['t'].to_numpy()
    covariate_cols = [col for col in df.columns if col.startswith('x')]
    x_df = df[covariate_cols]
    idx = df.index
    
    if t_max is None:
        t_max = np.max(t_obs)
    
    t_vals = np.arange(t_max + 1)
    
    m_vals_list = []
    x_aug_base = x_df.copy()
    outcome_model_cols = list(x_aug_base.columns) + ['t'] 
    for t in t_vals:
        x_aug = x_aug_base.copy()
        x_aug['t'] = t
        m_vals_list.append(outcome_model.predict(x_aug[outcome_model_cols]))
    m_vals = np.column_stack(m_vals_list)

    pscore_raw = pscore_model.predict_proba(x_df)
    n, _ = pscore_raw.shape
    pscore = np.zeros((n, len(t_vals)))
    
    for i, t_class in enumerate(pscore_model.classes_):
        if t_class < len(t_vals):
            pscore[:, t_class] = pscore_raw[:, i]

    indicator = (t_obs[:,None] == t_vals[None, :]).astype(float)
    
    influence = indicator / np.clip(pscore, 1e-12, None) * (y[:, None] - m_vals) + m_vals
    
    influence_df = pd.DataFrame(influence, index=idx, columns=[f"t_{t}" for t in t_vals])
    
    return influence_df

# def estimate_ate_aipw(df, outcome_model, pscore_model):
#     """
#     Binary treatment version of AIPW estimator
#     """
#     X = df[['x']].to_numpy()
#     Y = df['y'].to_numpy()
#     T = df['t'].to_numpy()
    
#     # Estimate outcome model E[Y|X,T]
#     X0 = np.column_stack([X, np.zeros_like(T)])
#     X1 = np.column_stack([X, np.ones_like(T)])
#     m0_hat = outcome_model.predict(X0)
#     m1_hat = outcome_model.predict(X1)
    
#     # Estimate propensity score P(T=1|X)
#     p_hat = pscore_model.predict_proba(X)[:, 1]
    
#     # AIPW formula
#     influence = (T / p_hat) * (Y - m1_hat) + m1_hat + ((1 - T) / (1 - p_hat)) * (Y - m0_hat) + m0_hat - (m1_hat - m0_hat)
    
#     # Calculate estimate, SE, and CI
#     ate = np.mean(influence)
#     se = np.std(influence, ddof=1) / np.sqrt(len(df))
#     ci = (ate - 1.96 * se, ate + 1.96 * se)
    
#     return ate, se, ci


# def analytical_variance(df, m_tx, p_x, weights):
#     """
#     Estimates the analytical variance.
#     """
#     aipw_matrix = estimate_aipw_matrix(df, m_tx, p_x).values
#     T_max = aipw_matrix.shape[1] - 1
    
#     # Compute the c vector
#     c = np.zeros(T_max + 1)
#     c[0] = -weights[0]
#     c[1:T_max] = weights[:-1] - weights[1:]
#     c[T_max] = weights[T_max-1] # This seems like an error in the image. Should be weights[T_max-1]? or [T_max]?
    
#     psi = aipw_matrix @ c
#     var_hat = np.var(psi, ddof=1) / len(df)
    
#     return var_hat


# def estimate_weighted_increments(df, m_tx, p_x):
#     """
#     Estimates weighted increments between AIPW estimates for each value of t.
    
#     Steps:
#     1. Computes empirical proportions of each value of t (used as weights)
#     2. Computes AIPW influence function matrix
#     3. Computes difference in mean influence across consecutive t values
#     4. Returns weighted differences using empirical t distribution
#     """
#     # Step 1: Compute empirical weights
#     t_vals, counts = np.unique(df['t'], return_counts=True)
#     max_t = np.max(t_vals)
#     weights = np.zeros(max_t + 1)
#     weights[t_vals] = counts / len(df)

#     # Step 2: Compute AIPW matrix
#     aipw_matrix = estimate_aipw_matrix(df, m_tx, p_x)
    
#     # Step 3: Compute consecutive mean differences
#     column_names = [f"f_t_{t}" for t in range(max_t + 1)]
#     means = aipw_matrix[column_names].mean(axis=0).to_numpy()
#     differences = np.diff(means) # length = max_t
    
#     # Step 4: Apply weights (set last weight to zero)
#     effective_weights = weights[:-1].copy()
#     effective_weights[-1] = 0.0 # Set weight for final bin to zero
#     weighted_differences = effective_weights * differences
#     estimate = weighted_differences.sum()
    
#     # Variance
#     variance = analytical_variance(df, m_tx, p_x, effective_weights)
    
#     # Return results
#     return {
#         'estimate': estimate,
#         'variance': variance,
#         'ci': (
#             estimate - 1.96 * np.sqrt(variance),
#             estimate + 1.96 * np.sqrt(variance)
#         ),
#         'weighted_differences': weighted_differences,
#         'empirical_weights': weights,
#         'differences': differences
#     }
    
# def estimate_aipw_matrix(df, m_tx, p_x):
#     """
#     Computes the AIPW influence function matrix for multiple treatment values
#     """
#     n = len(df)
#     t_vals = np.unique(df['t'])
#     y = df['y'].to_numpy()
#     t_obs = df['t'].to_numpy()
#     pscore_raw = p_x
#     pscore = np.zeros((n, len(t_vals)))

#     # probability for each treatment class
#     for i, t in enumerate(t_vals):
#         if i < pscore_raw.shape[1]:
#             pscore[:, i] = pscore_raw[:, i]

#     # safety check
#     if np.any(pscore.sum(axis=1) == 0):
#         raise ValueError("Some rows have zero probability across all classes!")

#     indicator = (t_obs[:, None] == t_vals[None, :]).astype(float)

#     # AIPW formula
#     influence = indicator / np.clip(pscore, 1e-12, None) * (y[:, None] - m_tx) + m_tx
#     influence_df = pd.DataFrame(influence, index=df.index, columns=[f"τ_{t}" for t in t_vals])
#     return influence_df


def estimate_ate_aipw(df, outcome_model, pscore_model):
    """
    Binary treatment version of AIPW estimator
    """
    X = df[['x']].to_numpy()
    T = df['t'].to_numpy()
    Y = df['y'].to_numpy()

    # outcome regression
    X0 = np.column_stack([X, np.zeros_like(T)])
    X1 = np.column_stack([X, np.ones_like(T)])
    mu0 = outcome_model.predict(X0)
    mu1 = outcome_model.predict(X1)

    # propensity
    p = pscore_model.predict_proba(X)[:, 1]

    # AIPW influence function
    influence = (
        (T / p) * (Y - mu1) + mu1
        - ((1 - T) / (1 - p)) * (Y - mu0) - mu0
    )

    ate = np.mean(influence)
    se = np.std(influence, ddof=1) / np.sqrt(len(df))
    ci = (ate - 1.96 * se, ate + 1.96 * se)
    return ate, se, ci


def analytical_variance(df, m_tx, p_x, weights):
    """
    Computes analytical variance for AIPW estimates
    """
    aipw_matrix = estimate_aipw_matrix(df, m_tx, p_x).values
    T_max = aipw_matrix.shape[1] - 1

    # compute c vector
    c = np.zeros(T_max + 1)
    c[0] = -weights[0]
    c[1:T_max] = weights[:-1] - weights[1:]
    c[T_max] = weights[T_max - 1]

    # variance
    psi = aipw_matrix @ c
    var_hat = np.var(psi, ddof=1) / len(df)
    return var_hat


def estimate_weighted_increments(df, m_tx, p_x):
    """
    Estimates weighted increments between AIPW estimates for each value of t.
    """
    t_vals, counts = np.unique(df['t'], return_counts=True)
    max_t = np.max(t_vals)
    weights = np.zeros(max_t + 1)
    weights[t_vals] = counts / len(df)

    # Step 2: Compute AIPW matrix
    aipw_matrix = estimate_aipw_matrix(df, m_tx, p_x)

    # Step 3: Compute consecutive mean differences
    column_names = [f"t_{t}" for t in range(max_t + 1)]
    means = aipw_matrix[column_names].mean(axis=0).to_numpy()
    differences = np.diff(means)

    # Step 4: Apply weights
    effective_weights = weights[:-1].copy()
    effective_weights[-1] = 0.0
    weighted_differences = effective_weights * differences
    estimate = weighted_differences.sum()

    variance = analytical_variance(df, m_tx, p_x, effective_weights)
    return (
        estimate,
        np.sqrt(variance),
        (
            estimate - 1.96 * np.sqrt(variance),
            estimate + 1.96 * np.sqrt(variance)
        )
    )

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

