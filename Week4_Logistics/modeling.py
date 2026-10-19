"""Delivery-duration regression at warehouse release; all records are fictional."""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

NUMERIC = ['distance_km', 'shipment_units', 'weight_kg', 'warehouse_dwell_hours']
CATEGORICAL = ['region', 'service']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'delivery_days'


def features(df):
    """Whitelist predictors: never infer features by dropping only the target."""
    if not set(FEATURES).issubset(df.columns):
        raise ValueError('Required prediction features missing')
    if df.empty:
        raise ValueError('Empty prediction input')
    x = df[FEATURES].copy()
    for col in NUMERIC:
        x[col] = pd.to_numeric(x[col], errors='raise')
        if np.isinf(x[col]).any() or (x[col].dropna() < 0).any():
            raise ValueError('Negative or infinite numeric input')
    return x


def chronological_splits(df):
    """Expanding monthly folds, purged until labels would have been available."""
    folds, rows = [], []
    for month in [3, 4, 5]:
        start = pd.Timestamp(2026, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        train = np.flatnonzero((df.dispatch_date < start) & (df.delivery_date < start))
        valid = np.flatnonzero((df.dispatch_date >= start) & (df.dispatch_date < end))
        if not len(train) or not len(valid):
            raise ValueError('Insufficient data for monthly cross-validation')
        folds.append((train, valid))
        rows.append({'validation_month': start.strftime('%Y-%m'), 'train_n': len(train),
                     'validation_n': len(valid),
                     'purged_n': int(((df.dispatch_date < start) & (df.delivery_date >= start)).sum()),
                     'latest_training_delivery': str(df.iloc[train].delivery_date.max().date())})
    cutoff = pd.Timestamp('2026-06-01')
    train = np.flatnonzero((df.dispatch_date < cutoff) & (df.delivery_date < cutoff))
    test = np.flatnonzero((df.dispatch_date >= cutoff) & (df.dispatch_date < pd.Timestamp('2026-07-01')))
    if not len(train) or not len(test):
        raise ValueError('Insufficient final train/test data')
    return folds, train, test, pd.DataFrame(rows)


def pipeline(estimator):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    category = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return Pipeline([('prepare', ColumnTransformer([('num', numeric, NUMERIC),
                                                    ('cat', category, CATEGORICAL)])),
                     ('model', estimator)])


def candidates():
    """Small, declared grid; seeds fixed and forest limited to one CPU."""
    yield 'Dummy mean', {}, pipeline(DummyRegressor(strategy='mean'))
    yield 'Linear regression', {}, pipeline(LinearRegression())
    for depth in [3, 6, 10]:
        for leaf in [5, 20]:
            params = {'max_depth': depth, 'min_samples_leaf': leaf}
            yield 'Decision tree', params, pipeline(DecisionTreeRegressor(**params, random_state=2026))
    for depth in [6, None]:
        for leaf in [2, 8]:
            params = {'n_estimators': 100, 'max_depth': depth, 'min_samples_leaf': leaf}
            yield 'Random forest', params, pipeline(RandomForestRegressor(**params, random_state=2026, n_jobs=1))


def metrics(y, prediction):
    """Metrics use continuous, unrounded predictions in days."""
    return {'mae_days': float(mean_absolute_error(y, prediction)),
            'rmse_days': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction))}


def train_compare(df):
    x, y = features(df), df[TARGET]
    folds, train, test, fold_info = chronological_splits(df)
    records, fold_scores, templates = [], [], {}
    for number, (name, params, estimator) in enumerate(candidates()):
        scores = []
        for k, (a, b) in enumerate(folds, 1):
            fitted = clone(estimator).fit(x.iloc[a], y.iloc[a])
            score = metrics(y.iloc[b], fitted.predict(x.iloc[b]))
            scores.append(score['mae_days'])
            fold_scores.append({'candidate_id': number, 'model': name, 'fold': k, **score})
        templates[number] = estimator
        records.append({'candidate_id': number, 'model': name, 'parameters': json.dumps(params),
                        'cv_mae_days': float(np.mean(scores)), 'cv_mae_sd': float(np.std(scores, ddof=0))})
    cv = pd.DataFrame(records).sort_values(['cv_mae_days', 'candidate_id']).reset_index(drop=True)
    winner = int(cv.iloc[0].candidate_id)
    # Freeze selection BEFORE inspecting June scores.
    best = clone(templates[winner]).fit(x.iloc[train], y.iloc[train])
    test_rows = []
    for name, group in cv.groupby('model', sort=False):
        candidate = int(group.iloc[0].candidate_id)
        fitted = best if candidate == winner else clone(templates[candidate]).fit(x.iloc[train], y.iloc[train])
        test_rows.append({'model': name, 'candidate_id': candidate,
                          **metrics(y.iloc[test], fitted.predict(x.iloc[test]))})
    return best, cv, pd.DataFrame(fold_scores), fold_info, pd.DataFrame(test_rows), train, test
