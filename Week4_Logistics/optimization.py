"""Hypothetical optimization: predicted tardiness, NOT observed causal savings."""
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix
from modeling import features


def allocate(cost, benefit, dates, budget, daily_cap):
    """Minimize incremental spend minus modeled tardiness-loss reduction.

    Each binary decision upgrades one eligible shipment. Daily cap counts upgrades,
    not total Express traffic. Zero decisions are always feasible.
    """
    cost, benefit = np.asarray(cost, float), np.asarray(benefit, float)
    dates = np.asarray(dates)
    if len(cost) != len(benefit) or len(cost) != len(dates):
        raise ValueError('Input lengths differ')
    if not np.isfinite(cost).all() or not np.isfinite(benefit).all() or (cost <= 0).any():
        raise ValueError('Costs must be positive and all values finite')
    if not np.isfinite(budget) or budget < 0 or not isinstance(daily_cap, (int, np.integer)) or daily_cap < 0:
        raise ValueError('Invalid budget or capacity')
    choice = np.zeros(len(cost), dtype=int)
    eligible = np.flatnonzero(benefit > cost + 1e-9)
    if not len(eligible) or budget == 0 or daily_cap == 0:
        return choice, {'success': True, 'status': 'No profitable feasible action', 'mip_gap': 0.0}
    c, b, d = cost[eligible], benefit[eligible], dates[eligible]
    days = np.unique(d)
    matrix = np.vstack([c] + [(d == day).astype(float) for day in days])
    upper = np.array([budget] + [daily_cap] * len(days), dtype=float)
    result = milp(c=c-b, integrality=np.ones(len(c)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(csr_matrix(matrix), np.full(len(upper), -np.inf), upper),
                  options={'time_limit': 60, 'mip_rel_gap': 0.0})
    if not result.success:
        raise RuntimeError(f'Allocation solver failed: {result.message}')
    choice[eligible] = np.rint(result.x).astype(int)
    if np.dot(cost, choice) > budget + 1e-5:
        raise AssertionError('Budget constraint violated')
    for day in np.unique(dates):
        if choice[dates == day].sum() > daily_cap:
            raise AssertionError('Daily capacity violated')
    return choice, {'success': bool(result.success), 'status': result.message,
                    'mip_gap': float(result.mip_gap)}


def greedy(cost, benefit, dates, budget, daily_cap):
    """Feasible comparator sorted by net predicted benefit per extra INR."""
    cost, benefit, dates = np.asarray(cost), np.asarray(benefit), np.asarray(dates)
    selected = np.zeros(len(cost), int); spend = 0.; counts = {}
    for i in np.argsort(-(benefit-cost)/cost, kind='stable'):
        if benefit[i] <= cost[i] or spend + cost[i] > budget + 1e-9:
            continue
        if counts.get(dates[i], 0) >= daily_cap:
            continue
        selected[i] = 1; spend += cost[i]; counts[dates[i]] = counts.get(dates[i], 0) + 1
    return selected


def scenarios(model, june):
    """Counterfactual feature substitution; an association is not a causal effect."""
    frame = june.copy().reset_index(drop=True)
    base = np.maximum(1., model.predict(features(frame)))
    changed = frame.copy(); changed['service'] = 'Express'
    upgraded = np.maximum(1., model.predict(features(changed)))
    promise = frame.promised_days.to_numpy()
    base_late = np.maximum(0., base - promise)
    extra = 200. + 2. * frame.distance_km.to_numpy()
    gain = np.maximum(0., base_late - np.maximum(0., upgraded - promise))
    standard = frame.service.to_numpy() == 'Standard'
    dates = frame.dispatch_date.dt.strftime('%Y-%m-%d').to_numpy()
    summary = []
    main_allocation = None
    for budget, penalty, cap in [(0., 1500., 4), (15000., 1500., 4), (30000., 1500., 4),
                                  (60000., 1500., 4), (30000., 750., 4), (30000., 3000., 4),
                                  (30000., 1500., 2), (30000., 1500., 8)]:
        idx = np.flatnonzero(standard)
        chosen, solver = allocate(extra[idx], penalty*gain[idx], dates[idx], budget, cap)
        selected = np.zeros(len(frame), int); selected[idx] = chosen
        heur = greedy(extra[idx], penalty*gain[idx], dates[idx], budget, cap)
        spend = float(extra @ selected)
        avoided = float(gain @ selected)
        heuristic_net = float((penalty*gain[idx]-extra[idx]) @ heur)
        net = penalty*avoided-spend
        assert net >= heuristic_net - .01 and net >= -.01
        prediction = np.where(selected == 1, upgraded, base)
        summary.append({'budget_inr': budget, 'penalty_inr_per_day': penalty, 'daily_upgrade_cap': cap,
                        'upgrades': int(selected.sum()), 'extra_spend_inr': spend,
                        'predicted_tardiness_days_before': float(base_late.sum()),
                        'predicted_tardiness_days_after': float(np.maximum(0., prediction-promise).sum()),
                        'predicted_days_avoided': avoided, 'net_modeled_benefit_inr': net,
                        'greedy_net_benefit_inr': heuristic_net, 'solver_success': solver['success'],
                        'mip_gap': solver['mip_gap']})
        if budget == 30000 and penalty == 1500 and cap == 4:
            main_allocation = frame[['shipment_id','dispatch_date','region','service','distance_km','promised_days']].copy()
            main_allocation['predicted_existing_days'] = base
            main_allocation['predicted_express_days'] = upgraded
            main_allocation['upgrade_selected'] = selected
            main_allocation['extra_cost_inr'] = extra*selected
            main_allocation['predicted_days_avoided'] = gain*selected
            main_allocation['predicted_after_days'] = prediction
    return pd.DataFrame(summary), main_allocation
