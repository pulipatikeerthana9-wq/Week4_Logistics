"""Fictional logistics simulation. No measured company data or calibrated rates."""
import numpy as np
import pandas as pd

SEED = 2026
N_SHIPMENTS = 3000
REGIONS = ['North', 'South', 'East', 'West']
SERVICES = ['Standard', 'Express']
ASSUMPTIONS = {
    'status': 'Entirely simulated; parameters are illustrative, not market estimates',
    'seed': SEED, 'shipments': N_SHIPMENTS,
    'period': '2026-01-01 to 2026-06-30 dispatch dates', 'currency': 'INR',
    'monthly_sampling_weights': [1.0, 1.0, 1.1, 1.2, 1.4, 1.6],
    'region_probabilities': [0.28, 0.26, 0.22, 0.24],
    'express_probability': 0.30,
    'distance_km': 'Gamma(shape=3, scale=170), clipped 40 to 1800; rounded 1 decimal',
    'units': 'Integer uniform 5 to 100 inclusive',
    'weight_kg': 'units * Uniform(1,4), rounded 2 decimals',
    'dwell_hours': 'Gamma(shape=2, scale=4) + 12 in East + 2 in North',
    'promised_days': 'ceil(distance/600) + 1 for Standard; ceil(distance/900) for Express',
    'transit_days': 'max(1, ceil(distance/speed + dwell/24 + Normal(0.15,0.45))); speed 500 Standard, 850 Express',
    'cost_fixed_inr': '300 Standard or 500 Express',
    'cost_distance_inr': 'distance * 4 Standard or * 6 Express',
    'cost_weight_inr': 'weight * 2',
    'cost_handling_inr': 'dwell * 8',
    'cost_variation_inr': 'Uniform(0,150)',
    'total_cost': 'Sum of five rounded components; no warehouse inventory, fuel market, VAT or profit model',
    'limitations': 'All relationships reflect the programmed scenario; dates have no observed seasonality; rates are assumptions; no cancellations or split shipments.'
}


def simulate(n=N_SHIPMENTS, seed=SEED):
    if not isinstance(n, int) or n < 1:
        raise ValueError('n must be a positive integer')
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2026-01-01', '2026-06-30')
    weights = np.array(ASSUMPTIONS['monthly_sampling_weights'])[dates.month - 1]
    dispatch = pd.DatetimeIndex(rng.choice(dates, n, p=weights/weights.sum()))
    region = rng.choice(REGIONS, n, p=[.28,.26,.22,.24])
    service = rng.choice(SERVICES, n, p=[.7,.3])
    express = service == 'Express'
    distance = np.round(np.clip(rng.gamma(3,170,n),40,1800),1)
    units = rng.integers(5,101,n)
    weight = np.round(units*rng.uniform(1,4,n),2)
    dwell = np.round(rng.gamma(2,4,n)+12*(region=='East')+2*(region=='North'),2)
    promised = np.ceil(distance/np.where(express,900,600)).astype(int)+(~express).astype(int)
    transit = np.maximum(1,np.ceil(distance/np.where(express,850,500)+dwell/24+rng.normal(.15,.45,n))).astype(int)
    df = pd.DataFrame({
        'shipment_id': [f'SIM{i:05d}' for i in range(1,n+1)],
        'dispatch_date':dispatch, 'region':region, 'service':service,
        'distance_km':distance, 'shipment_units':units, 'weight_kg':weight,
        'warehouse_dwell_hours':dwell, 'promised_days':promised,
        'delivery_days':transit, 'delivery_date':dispatch+pd.to_timedelta(transit,unit='D'),
        'cost_fixed_inr':np.where(express,500.,300.),
        'cost_distance_inr':np.round(distance*np.where(express,6,4),2),
        'cost_weight_inr':np.round(weight*2,2),
        'cost_handling_inr':np.round(dwell*8,2),
        'cost_variation_inr':np.round(rng.uniform(0,150,n),2)})
    components = [c for c in df if c.startswith('cost_')]
    df['transport_cost_inr'] = df[components].sum(axis=1).round(2)
    df['delay_days'] = df.delivery_days-df.promised_days
    df['late_flag'] = (df.delay_days>0).astype(int)
    validate(df)
    return df.sort_values(['dispatch_date','shipment_id']).reset_index(drop=True)


def validate(df):
    required = ['shipment_id','dispatch_date','delivery_date','region','service','distance_km',
                'shipment_units','weight_kg','warehouse_dwell_hours','delivery_days','promised_days',
                'delay_days','late_flag','transport_cost_inr','cost_fixed_inr','cost_distance_inr',
                'cost_weight_inr','cost_handling_inr','cost_variation_inr']
    if not set(required).issubset(df.columns):
        raise ValueError('Required shipment columns missing')
    if df.empty or df[required].isna().any().any() or not df.shipment_id.is_unique:
        raise ValueError('Empty data, missing values or duplicate shipment IDs')
    if not df.region.isin(REGIONS).all() or not df.service.isin(SERVICES).all():
        raise ValueError('Unknown region or service')
    positive = ['distance_km','shipment_units','weight_kg','delivery_days','promised_days','transport_cost_inr']
    if (df[positive]<=0).any().any() or (df.warehouse_dwell_hours<0).any():
        raise ValueError('Invalid nonpositive measurement or negative dwell')
    integer_cols = ['shipment_units','delivery_days','promised_days']
    if not (df[integer_cols] % 1 == 0).all().all():
        raise ValueError('Units and transit days must be integers')
    if not ((df.delivery_date-df.dispatch_date).dt.days==df.delivery_days).all():
        raise ValueError('Delivery dates do not reconcile')
    if not (df.delay_days==df.delivery_days-df.promised_days).all():
        raise ValueError('Delay mismatch')
    if not (df.late_flag==(df.delay_days>0).astype(int)).all():
        raise ValueError('Late flag mismatch')
    components = ['cost_fixed_inr','cost_distance_inr','cost_weight_inr','cost_handling_inr','cost_variation_inr']
    if (df[components]<0).any().any() or not np.allclose(df[components].sum(axis=1),df.transport_cost_inr,atol=.001):
        raise ValueError('Cost components do not reconcile')
