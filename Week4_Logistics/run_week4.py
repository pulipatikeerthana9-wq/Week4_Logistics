"""Run: python run_week4.py. Rebuilds analysis outputs, not the supplied Word file."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from simulation import simulate, ASSUMPTIONS
from modeling import train_compare, features, metrics
from optimization import scenarios

ROOT = Path(__file__).resolve().parent


def main():
    data, results = ROOT/'data', ROOT/'results'
    figures = results/'figures'
    data.mkdir(exist_ok=True); results.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    df = simulate()
    df.to_csv(data/'simulated_shipments.csv', index=False)
    (data/'simulation_assumptions.json').write_text(json.dumps(ASSUMPTIONS, indent=2), encoding='utf-8')
    model, cv, folds, split_info, scores, train, test = train_compare(df)
    for filename, table in [('cv_candidates.csv', cv), ('cv_fold_scores.csv', folds),
                            ('validation_splits.csv', split_info), ('test_model_metrics.csv', scores)]:
        table.to_csv(results/filename, index=False)
    june = df.iloc[test].copy()
    predictions = model.predict(features(june))
    june['prediction_days'] = predictions
    june['residual_days'] = june.delivery_days - predictions
    june.to_csv(results/'test_predictions.csv', index=False)
    subgroup = []
    for column in ['region', 'service']:
        for label, subset in june.groupby(column):
            subgroup.append({'group_by': column, 'group': label, 'n': len(subset),
                             **metrics(subset.delivery_days, subset.prediction_days),
                             'mean_residual_days': float(subset.residual_days.mean())})
    pd.DataFrame(subgroup).to_csv(results/'subgroup_metrics.csv', index=False)
    importance = permutation_importance(model, features(june), june.delivery_days,
                                        scoring='neg_mean_absolute_error', n_repeats=10, random_state=2026, n_jobs=1)
    imp = pd.DataFrame({'feature': features(june).columns, 'mae_increase_days': importance.importances_mean,
                        'repeat_sd': importance.importances_std}).sort_values('mae_increase_days', ascending=False)
    imp.to_csv(results/'permutation_importance.csv', index=False)
    scenario, allocation = scenarios(model, df.iloc[test])
    scenario.to_csv(results/'optimization_sensitivity.csv', index=False)
    allocation.to_csv(results/'upgrade_allocation.csv', index=False)
    joblib.dump(model, results/'delivery_model.joblib')
    check = joblib.load(results/'delivery_model.joblib').predict(features(june))
    assert np.allclose(check, predictions)
    summary = {'study_status': 'Entirely hypothetical simulation; not company performance',
               'n_shipments': len(df), 'train_n': len(train), 'test_n': len(test),
               'purged_pre_june_n': int(((df.dispatch_date < pd.Timestamp('2026-06-01')) &
                                          (df.delivery_date >= pd.Timestamp('2026-06-01'))).sum()),
               'selected_model': cv.iloc[0]['model'], 'selected_parameters': json.loads(cv.iloc[0]['parameters']),
               'cv_mae_days': float(cv.iloc[0]['cv_mae_days']),
               'holdout_metrics': metrics(june.delivery_days, predictions),
               'within_one_day_fraction': float((np.abs(june.residual_days) <= 1).mean()),
               'p90_absolute_error_days': float(np.quantile(np.abs(june.residual_days), .90)),
               'mean_residual_days': float(june.residual_days.mean()),
               'optimization': scenario.iloc[2].to_dict()}
    (results/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    environment = {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                   for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','joblib']}}
    (results/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in list(ROOT.glob('*.py')) + [data/'simulated_shipments.csv']}
    (results/'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')
    plot_results(scores, june, imp, scenario, figures)
    print(json.dumps(summary, indent=2))
    print('SUCCESS: Week 4 models, validation and hypothetical optimization completed.')


def plot_results(scores, june, importance, scenarios_df, folder):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8,4))
    ordered = scores.sort_values('mae_days', ascending=False)
    ax.barh(ordered.model, ordered.mae_days, color='#466D8A')
    ax.set_xlim(left=0); ax.set_xlabel('June holdout MAE (days; lower is better)')
    ax.set_title('Model comparison — hypothetical shipment data')
    fig.tight_layout(); fig.savefig(folder/'01_model_comparison.png',dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    axes[0].scatter(june.delivery_days, june.prediction_days, s=13, alpha=.25, color='#466D8A')
    high=max(june.delivery_days.max(),june.prediction_days.max())+.2
    axes[0].plot([0,high],[0,high],'k--',linewidth=1); axes[0].set(xlabel='Actual simulated days',ylabel='Predicted days',title='Actual vs. predicted')
    axes[1].hist(june.residual_days,bins=25,color='#466D8A'); axes[1].axvline(0,color='black',linestyle='--')
    axes[1].set(xlabel='Actual minus predicted (days)',ylabel='Shipments',title='June residual distribution')
    fig.tight_layout(); fig.savefig(folder/'02_diagnostics.png',dpi=200); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.barh(importance.feature[::-1],importance.mae_increase_days[::-1],color='#466D8A')
    ax.set(xlabel='Mean MAE increase after permutation (days)',title='Holdout feature importance — not causal effects')
    fig.tight_layout(); fig.savefig(folder/'03_importance.png',dpi=200); plt.close(fig)
    sub=scenarios_df.iloc[:4]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(sub.budget_inr,sub.predicted_days_avoided,'o-',color='#466D8A')
    ax.set(xlabel='Monthly extra-spend budget (hypothetical INR)',ylabel='Predicted tardiness days avoided',title='Budget sensitivity — penalty INR 1,500/day, cap 4/day')
    ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(folder/'04_budget_sensitivity.png',dpi=200); plt.close(fig)


if __name__ == '__main__': main()
