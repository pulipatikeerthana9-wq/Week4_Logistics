import unittest
import numpy as np
from simulation import simulate, validate
from modeling import features, chronological_splits, pipeline, metrics, FEATURES
from sklearn.linear_model import LinearRegression
from optimization import allocate, greedy, scenarios
import itertools


class DataModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = simulate()

    def test_seed_reproducible(self):
        self.assertTrue(simulate(100).equals(simulate(100)))

    def test_invalid_simulation(self):
        with self.assertRaises(ValueError): simulate(0)

    def test_duplicates_fail(self):
        bad = self.df.copy()
        bad.loc[1, 'shipment_id'] = bad.loc[0, 'shipment_id']
        with self.assertRaises(ValueError): validate(bad)

    def test_feature_whitelist(self):
        self.assertEqual(list(features(self.df)), FEATURES)
        self.assertNotIn('delivery_date', FEATURES)
        self.assertNotIn('transport_cost_inr', FEATURES)
        self.assertNotIn('late_flag', FEATURES)
        self.assertNotIn('promised_days', FEATURES)

    def test_label_purge(self):
        folds, train, test, info = chronological_splits(self.df)
        for (a, b), month in zip(folds, [3, 4, 5]):
            self.assertTrue((self.df.iloc[a].delivery_date < self.df.iloc[b].dispatch_date.min()).all())
            self.assertTrue((self.df.iloc[b].dispatch_date.dt.month == month).all())
            self.assertEqual(len(set(a) & set(b)), 0)
        self.assertTrue((self.df.iloc[train].delivery_date < np.datetime64('2026-06-01')).all())
        self.assertTrue((self.df.iloc[test].dispatch_date.dt.month == 6).all())

    def test_missing_feature_fails(self):
        with self.assertRaises(ValueError): features(self.df.drop(columns='distance_km'))

    def test_negative_feature_fails(self):
        bad = self.df.copy(); bad.loc[0, 'weight_kg'] = -1
        with self.assertRaises(ValueError): features(bad)

    def test_insufficient_dates(self):
        with self.assertRaises(ValueError): chronological_splits(self.df.head(2))

    def test_hand_metrics(self):
        m = metrics([1, 2, 3], [1, 2, 2])
        self.assertAlmostEqual(m['mae_days'], 1 / 3)
        self.assertAlmostEqual(m['rmse_days'], np.sqrt(1 / 3))
        self.assertAlmostEqual(m['r2'], .5)

    def test_unseen_category_and_missing_numeric(self):
        x = features(self.df)
        fitted = pipeline(LinearRegression()).fit(x.iloc[:100], self.df.delivery_days.iloc[:100])
        sample = x.iloc[:1].copy(); sample['region'] = 'New'; sample['weight_kg'] = np.nan
        self.assertTrue(np.isfinite(fitted.predict(sample)).all())


class OptimizationTests(unittest.TestCase):
    def test_brute_force_optimum(self):
        cost = np.array([4., 5., 3., 6.]); benefit = np.array([9., 12., 8., 13.])
        dates = np.array(['a', 'a', 'b', 'b'])
        selected, status = allocate(cost, benefit, dates, 10., 1)
        possible = []
        for bits in itertools.product([0, 1], repeat=4):
            z = np.array(bits)
            if cost @ z <= 10 and z[:2].sum() <= 1 and z[2:].sum() <= 1:
                possible.append((benefit-cost) @ z)
        self.assertAlmostEqual((benefit-cost) @ selected, max(possible))
        self.assertTrue(status['success'])

    def test_zero_budget(self):
        z, _ = allocate([10], [100], ['a'], 0., 1)
        self.assertEqual(z.sum(), 0)

    def test_zero_capacity(self):
        z, _ = allocate([10], [100], ['a'], 100., 0)
        self.assertEqual(z.sum(), 0)

    def test_no_profitable_action(self):
        z, _ = allocate([10, 20], [9, 20], ['a', 'b'], 100., 2)
        self.assertEqual(z.sum(), 0)

    def test_empty_candidates(self):
        z, _ = allocate([], [], [], 100., 2)
        self.assertEqual(len(z), 0)

    def test_invalid_cost(self):
        with self.assertRaises(ValueError): allocate([-1], [5], ['a'], 100., 2)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError): allocate([1], [5], ['a'], -1., 2)

    def test_invalid_capacity(self):
        with self.assertRaises(ValueError): allocate([1], [5], ['a'], 100., 1.5)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError): allocate([1, 2], [5], ['a'], 100., 2)

    def test_optimum_not_worse_than_greedy(self):
        c = np.array([4., 5., 6.]); b = np.array([10., 11., 20.]); d = np.array(['a']*3)
        z, _ = allocate(c, b, d, 10., 2)
        g = greedy(c, b, d, 10., 2)
        self.assertGreaterEqual((b-c)@z, (b-c)@g)

    def test_scenario_preserves_promises(self):
        df = simulate()
        train = df[df.dispatch_date.dt.month < 6]
        june = df[df.dispatch_date.dt.month == 6]
        model = pipeline(LinearRegression()).fit(features(train), train.delivery_days)
        original = june.copy()
        summary, plan = scenarios(model, june)
        self.assertTrue(june.equals(original))
        self.assertTrue(np.array_equal(plan.promised_days, june.promised_days))
        self.assertTrue((plan.loc[plan.upgrade_selected == 1, 'service'] == 'Standard').all())
        self.assertLessEqual(plan.extra_cost_inr.sum(), 30000.00001)
        self.assertLessEqual(plan.groupby('dispatch_date').upgrade_selected.sum().max(), 4)


if __name__ == '__main__': unittest.main()
