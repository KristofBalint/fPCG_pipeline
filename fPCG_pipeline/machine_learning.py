import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, \
    recall_score, precision_score, confusion_matrix
from imblearn.metrics import specificity_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
from typing import Optional, Union
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


warnings.filterwarnings("ignore", category=UserWarning)


def prepare_X_y(features_df,
                drop_target_columns,
                drop_meta_columns=None,
                required_features=None,
                outcome_def=None):
    """
    outcome_def: dict with:
        - 'target': column name in features_df
        - optional 'transform': callable to apply to the target series (returns categorical/labels)
        - optional 'dummy': name of dummy column to select after get_dummies (if multi-category)
    Returns: X (DataFrame), y (Series)
    """
    if drop_meta_columns is None:
        drop_meta_columns = []

    df = features_df.copy()
    # Drop columns that are meant to be removed (if present)
    to_drop = [c for c in drop_target_columns if c in df.columns]
    df_X = df.drop(
        columns=to_drop + [c for c in drop_meta_columns if c in df.columns],
        errors='ignore')

    # Convert bools to ints
    for column in df_X.select_dtypes(include=['bool']).columns:
        df_X[column] = df_X[column].astype(int)

    # Drop rows missing critical features (if provided)
    if required_features:
        existing_required = [c for c in required_features if c in df_X.columns]
        if existing_required:
            df_X = df_X.dropna(subset=existing_required)

    # Build Y from original features_df but align to the filtered X index
    idx = df_X.index
    if outcome_def is None:
        raise ValueError("outcome_def must be provided")

    target_col = outcome_def['target']
    y_source = features_df.loc[idx, target_col]

    # optional transformation
    if 'transform' in outcome_def and callable(outcome_def['transform']):
        y_trans = y_source.apply(outcome_def['transform'])
        dummies = pd.get_dummies(y_trans)
        if 'dummy' in outcome_def:
            if outcome_def['dummy'] in dummies.columns:
                y = dummies[outcome_def['dummy']]
            else:
                # fallback: pick first dummy and warn
                print(
                    f"Warning: dummy '{outcome_def['dummy']}' not found for target '{target_col}'. Using first dummy '{dummies.columns[0]}'.")
                y = dummies.iloc[:, 0]
        else:
            # if single-column result, return it
            if dummies.shape[1] == 1:
                y = dummies.iloc[:, 0]
            else:
                y = dummies.iloc[:, 0]  # fallback
    elif 'dummy' in outcome_def:
        dummies = pd.get_dummies(y_source, drop_first=False)
        if outcome_def['dummy'] in dummies.columns:
            y = dummies[outcome_def['dummy']]
        else:
            print(
                f"Warning: dummy '{outcome_def['dummy']}' not found for target '{target_col}'. Using first dummy '{dummies.columns[0]}'.")
            y = dummies.iloc[:, 0]
    else:
        y = y_source

    # ensure y is binary numeric if possible
    if y.dtype == 'O' or y.dtype.name.startswith('category'):
        try:
            y = pd.to_numeric(y)
        except:
            # if still non-numeric, map categories to integers
            y = pd.factorize(y)[0]

    return df_X, y


def train_and_evaluate(X, y, model_conf,
                       task_conf=None,
                       n_bootstraps=100,
                       global_use_smote=True,
                       test_size=0.2,
                       random_state_base=0,
                       mean_fpr_grid=None,
                       verbose=False):
    """
    model_conf: dict with keys:
        - 'estimator': sklearn-like estimator (unfitted)
        - optional 'skip_smote': True/False (if True, will avoid SMOTE unless task_conf forces it)
        - optional 'set_scale_pos_weight': True/False (if True and estimator supports set_params, sets XGBoost's scale_pos_weight)
    task_conf: dict with optional 'use_smote' (True/False) to *force* SMOTE on/off for this task

    Returns a results dict with ROC tprs/aucs and metric lists.
    """
    if mean_fpr_grid is None:
        mean_fpr_grid = np.linspace(0, 1, 100)

    tprs = []
    aucs = []
    metrics = {'balanced_accuracy': [], 'sensitivity': [], 'specificity': [],
               'precision': []}

    for i in range(n_bootstraps):
        rs = random_state_base + i
        # stratify when possible
        stratify = y if len(np.unique(y)) == 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=rs,
                                                            stratify=stratify)

        # bootstrap sample from training set
        X_boot, y_boot = resample(X_train, y_train, replace=True,
                                  n_samples=len(X_train), random_state=rs)

        # decide whether to use SMOTE:
        if task_conf and ('use_smote' in task_conf):
            use_smote = bool(task_conf['use_smote'])
        elif model_conf.get('skip_smote', False):
            use_smote = False
        else:
            use_smote = bool(global_use_smote)

        if use_smote:
            sm = SMOTE(random_state=rs)
            try:
                X_res, y_res = sm.fit_resample(X_boot, y_boot)
            except Exception as e:
                # fallback to original bootstrap if SMOTE fails
                print(
                    f"SMOTE failed at bootstrap {i}: {e}. Using bootstrap sample without SMOTE.")
                X_res, y_res = X_boot, y_boot
        else:
            X_res, y_res = X_boot, y_boot

        # handle special model settings (e.g., xgboost scale_pos_weight)
        estimator = clone(model_conf['estimator'])
        if model_conf.get('set_scale_pos_weight', False):
            # compute ratio on resampled labels
            n_pos = np.sum(y_res == 1)
            n_neg = np.sum(y_res == 0)
            if n_pos == 0:
                scale = 1.0
            else:
                scale = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
            # set if estimator accepts parameter
            try:
                estimator.set_params(scale_pos_weight=scale)
            except Exception:
                pass

        # fit
        estimator.fit(X_res, y_res)

        # get probability-like scores
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            score = estimator.decision_function(X_test)
            # convert to 0..1
            y_proba = (score - score.min()) / (score.max() - score.min() + 1e-9)
        else:
            raise ValueError(
                "Estimator has neither predict_proba nor decision_function. Provide a probability-capable model.")

        # ROC interpolation
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        interp_tpr = np.interp(mean_fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

        # binary prediction at 0.5
        y_pred = (y_proba > 0.5).astype(int)
        metrics['balanced_accuracy'].append(
            balanced_accuracy_score(y_test, y_pred))
        metrics['sensitivity'].append(
            recall_score(y_test, y_pred, zero_division=0))
        metrics['specificity'].append(specificity_score(y_test, y_pred))
        metrics['precision'].append(
            precision_score(y_test, y_pred, zero_division=0))
        if verbose:
            print(f"{rs}th bootstrap done")

    # summarize
    results = {
        'mean_fpr': mean_fpr_grid,
        'tprs': np.array(tprs),
        'aucs': np.array(aucs),
        'metrics': {k: np.array(v) for k, v in metrics.items()},
        'medians': {
            'median_tpr': np.median(tprs, axis=0),
            'q1_tpr': np.percentile(tprs, 25, axis=0),
            'q3_tpr': np.percentile(tprs, 75, axis=0),
            'median_auc': np.median(aucs),
            'median_metrics': {k: np.median(v) for k, v in metrics.items()}
        }
    }
    return results

def plot_perm_importance(
    estimator,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = -1,
    n_top: Optional[int] = None,
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    scoring: Optional[str] = None,
    fit_clone_if_unfitted: bool = False,
    fit_kwargs: Optional[dict] = None,
    title: Optional[str] = None,
    return_df: bool = False,
):
    """
    Compute permutation importance and plot a horizontal bar chart with error bars.

    Parameters
    ----------
    estimator :
        A fitted sklearn estimator (must support predict or predict_proba as used by scoring).
        If `fit_clone_if_unfitted=True`, a clone will be fitted on (X, y) if estimator is not fitted.
    X : DataFrame or ndarray
        Feature matrix (columns used as feature names if DataFrame).
    y : Series or ndarray
        Target labels.
    n_repeats : int
        Number of permutation repeats (passed to permutation_importance).
    random_state : int
        Random seed for permutation_importance.
    n_jobs : int
        Jobs for permutation_importance.
    n_top : Optional[int]
        If set, only plot the top-n features (by mean importance).
    figsize : tuple
        Figure size for matplotlib.
    ax : Optional[plt.Axes]
        If provided, draw on this Axes; otherwise create a new figure.
    scoring : str or callable, optional
        Scoring to use (passed to permutation_importance). If None, uses estimator default scorer.
    fit_clone_if_unfitted : bool
        If True and the estimator is not fitted, the function will fit a clone on (X, y) before computing importance.
        If False and estimator is not fitted, raises a NotFittedError.
    fit_kwargs : dict, optional
        Extra kwargs passed to estimator.fit when fitting the clone.
    title : str, optional
        Title for the plot. If None a default title is used.
    return_df : bool
        If True, return the importance dataframe in addition to plotting.

    Returns
    -------
    pd.DataFrame (optional)
        DataFrame containing Feature, Importance (mean), and Std columns (returned if return_df=True).
    """
    # Prepare fit kwargs
    fit_kwargs = fit_kwargs or {}

    # If X is DataFrame, preserve column names; otherwise generate names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # Ensure estimator is fitted (or fit a clone if requested)
    try:
        check_is_fitted(estimator)
        used_estimator = estimator  # already fitted; use as-is
    except (NotFittedError, Exception):
        if fit_clone_if_unfitted:
            used_estimator = clone(estimator)
            used_estimator.fit(X, y, **fit_kwargs)
        else:
            raise NotFittedError(
                "The provided estimator does not appear to be fitted. "
                "Either provide a fitted estimator or set fit_clone_if_unfitted=True."
            )

    # Compute permutation importance
    result = permutation_importance(
        used_estimator,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        scoring=scoring,
    )

    importances = result.importances_mean
    std = result.importances_std

    # Build DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Std": std
    })

    # Sort and optionally select top-n
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    if n_top is not None and n_top > 0:
        df_plot = df.head(n_top)
    else:
        df_plot = df

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    y_pos = np.arange(len(df_plot))
    # We want descending importance from top to bottom on the y-axis -> invert
    features_plot = df_plot["Feature"].values[::-1]
    importances_plot = df_plot["Importance"].values[::-1]
    std_plot = df_plot["Std"].values[::-1]

    ax.barh(y=features_plot, width=importances_plot, xerr=std_plot, capsize=4)
    ax.set_xlabel("Permutation importance (mean)")
    ax.set_ylabel("Feature")
    if title is None:
        title = f"Permutation feature importances (n_repeats={n_repeats})"
    ax.set_title(title)
    plt.tight_layout()

    if return_df:
        return df

