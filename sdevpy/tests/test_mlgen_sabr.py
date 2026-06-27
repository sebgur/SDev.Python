import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from sdevpy.volatility.mlsurfacegen.sabrgenerator import SabrGenerator, sabr_obj


SABR_PARAMS = {'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25}
_SABR_MOD = "sdevpy.volatility.mlsurfacegen.sabrgenerator"


class TestSabrGeneratorPriceStraddlesRefNvol:
    """Tests for the output_nvol=True branch of price_straddles_ref."""

    @patch(f"{_SABR_MOD}.bachelier.implied_vol_jaeckel")
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_output_nvol_shape(self, mock_sabr, mock_black, mock_jaeckel):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        mock_jaeckel.return_value = np.array([0.005])
        g = SabrGenerator(shift=0.0)
        expiries = np.array([0.5, 1.0])
        strikes = np.array([[0.02, 0.03], [0.02, 0.03]])
        result = g.price_straddles_ref(expiries, strikes, 0.025, SABR_PARAMS, output_nvol=True)
        assert result.shape == (2, 2)

    @patch(f"{_SABR_MOD}.bachelier.implied_vol_jaeckel")
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_output_nvol_value_from_jaeckel(self, mock_sabr, mock_black, mock_jaeckel):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        mock_jaeckel.return_value = np.array([0.0072])
        g = SabrGenerator(shift=0.0)
        result = g.price_straddles_ref(
            np.array([1.0]), np.array([[0.025]]), 0.025, SABR_PARAMS, output_nvol=True)
        assert result[0][0] == pytest.approx(0.0072)

    @patch(f"{_SABR_MOD}.bachelier.implied_vol_jaeckel")
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_output_nvol_exception_returns_sentinel(self, mock_sabr, mock_black, mock_jaeckel):
        """When implied_vol_jaeckel raises any exception the value must be -9999."""
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        mock_jaeckel.side_effect = RuntimeError("convergence failure")
        g = SabrGenerator(shift=0.0)
        result = g.price_straddles_ref(
            np.array([1.0]), np.array([[0.025]]), 0.025, SABR_PARAMS, output_nvol=True)
        assert result[0][0] == -9999

    @patch(f"{_SABR_MOD}.bachelier.implied_vol_jaeckel")
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_noise_applied_when_prob_is_one(self, mock_sabr, mock_black, mock_jaeckel):
        """noise_prob=1.0 always applies noise; control rand via mock."""
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        mock_jaeckel.return_value = np.array([0.005])
        g = SabrGenerator(shift=0.0)
        with patch("numpy.random.rand", side_effect=[0.0, 0.9]):
            # First call: gate (0.0 < 1.0 → noise on)
            # Second call: magnitude  noise = (0.9-0.5)*0.1/0.5 = 0.08
            result = g.price_straddles_ref(
                np.array([1.0]), np.array([[0.025]]), 0.025, SABR_PARAMS,
                output_nvol=True, rel_noise=0.1, noise_prob=1.0)
        expected = 0.005 * (1.0 + (0.9 - 0.5) * 0.1 / 0.5)
        assert result[0][0] == pytest.approx(expected)

    @patch(f"{_SABR_MOD}.bachelier.implied_vol_jaeckel")
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_no_noise_when_prob_is_zero(self, mock_sabr, mock_black, mock_jaeckel):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        mock_jaeckel.return_value = np.array([0.005])
        g = SabrGenerator(shift=0.0)
        result = g.price_straddles_ref(
            np.array([1.0]), np.array([[0.025]]), 0.025, SABR_PARAMS,
            output_nvol=True, rel_noise=0.5, noise_prob=0.0)  # prob=0 → never noised
        assert result[0][0] == pytest.approx(0.005)


class TestSabrGeneratorPriceStraddlesMod:
    """Tests for price_straddles_mod (entirely uncovered)."""

    def _make_gen_and_model(self, n_exp, n_str):
        g = SabrGenerator()
        model = MagicMock()
        model.predict.return_value = np.tile([0.2, 0.5, 0.55, -0.25], (n_exp, 1))
        g.price_straddles_ref = MagicMock(
            return_value=np.array([[0.002] * n_str]))
        return g, model

    def test_mod_params_and_prices_shape(self):
        n_exp, n_str = 2, 3
        g, model = self._make_gen_and_model(n_exp, n_str)
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        mkt_prices = np.ones((n_exp, n_str)) * 0.001
        mod_params, mod_prices = g.price_straddles_mod(model, expiries, strikes, 0.025, mkt_prices)
        assert len(mod_params) == n_exp
        assert mod_prices.shape == (n_exp, n_str)

    def test_mod_params_keys(self):
        n_exp, n_str = 2, 2
        g, model = self._make_gen_and_model(n_exp, n_str)
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        mkt_prices = np.ones((n_exp, n_str)) * 0.001
        mod_params, _ = g.price_straddles_mod(model, expiries, strikes, 0.025, mkt_prices)
        assert set(mod_params[0].keys()) == {'LnVol', 'Beta', 'Nu', 'Rho'}
        assert mod_params[0]['LnVol'] == pytest.approx(0.2)

    def test_model_receives_correct_number_of_columns(self):
        n_exp, n_str = 2, 3
        g, model = self._make_gen_and_model(n_exp, n_str)
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        mkt_prices = np.ones((n_exp, n_str)) * 0.001
        g.price_straddles_mod(model, expiries, strikes, 0.025, mkt_prices)
        # points = [expiry, fwd] + n_str market prices → n_str + 2 columns
        assert model.predict.call_args[0][0].shape == (n_exp, 2 + n_str)

    def test_output_nvol_forwarded_to_price_straddles_ref(self):
        g, model = self._make_gen_and_model(1, 2)
        expiries = np.array([1.0])
        strikes = np.array([[0.02, 0.03]])
        mkt_prices = np.array([[0.001, 0.001]])
        g.price_straddles_mod(model, expiries, strikes, 0.025, mkt_prices, output_nvol=True)
        # output_nvol is the 5th positional arg to price_straddles_ref
        assert g.price_straddles_ref.call_args[0][4] is True


class TestSabrGeneratorGenerateSamplesInverse:
    _RG = {'Ttm': [0.5, 2.0], 'F': [0.01, 0.04],
           'LnVol': [0.1, 0.3], 'Beta': [0.3, 0.7], 'Nu': [0.1, 0.5], 'Rho': [-0.5, 0.5]}

    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_output_columns_include_strike_headers(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        g = SabrGenerator(num_expiries=2, seed=0)
        spreads = [-10, 0, 10]
        df = g.generate_samples_inverse(2 * g.num_expiries, self._RG, spreads)
        for col in ['Ttm', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'K0', 'K1', 'K2']:
            assert col in df.columns

    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_number_of_strike_columns_matches_spreads(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        g = SabrGenerator(num_expiries=2, seed=0)
        spreads = [0, 50]
        df = g.generate_samples_inverse(2 * g.num_expiries, self._RG, spreads)
        assert 'K0' in df.columns and 'K1' in df.columns and 'K2' not in df.columns

    def test_use_nvol_cleansing_drops_out_of_range_rows(self):
        """Rows where any nvol > max_vol must be dropped by the cleansing step."""
        g = SabrGenerator(num_expiries=2, seed=0)
        spreads = [0]  # 1 strike column → K0

        # Surface mock: first expiry returns nvol=0.5 (> max_vol=0.2), second returns 0.01
        mock_psref = MagicMock(return_value=np.array([[0.5], [0.01]]))
        with patch.object(g, 'price_straddles_ref', mock_psref):
            df = g.generate_samples_inverse(
                2 * g.num_expiries, self._RG, spreads,
                use_nvol=True, max_vol=0.2, min_vol=1e-4)

        assert (df['K0'] <= 0.2).all()

    def test_use_nvol_cleansing_drops_below_min_vol(self):
        """Rows where any nvol < min_vol must be dropped."""
        g = SabrGenerator(num_expiries=2, seed=0)
        spreads = [0]

        mock_psref = MagicMock(return_value=np.array([[1e-6], [0.01]]))
        with patch.object(g, 'price_straddles_ref', mock_psref):
            df = g.generate_samples_inverse(
                2 * g.num_expiries, self._RG, spreads,
                use_nvol=True, max_vol=0.2, min_vol=1e-4)

        assert (df['K0'] >= 1e-4).all()


class TestSabrObj:
    """Tests for the module-level sabr_obj function."""

    @patch(f"{_SABR_MOD}.metrics.rmsew")
    def test_returns_scaled_rmsew(self, mock_rmsew):
        mock_rmsew.return_value = 0.5
        g = SabrGenerator(shift=0.0)
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.002, 0.002]]))
        x = [0.20, 0.5, 0.55, -0.25]
        result = sabr_obj(x, g, 1.0, np.array([0.02, 0.03]), 0.025,
                          np.array([0.002, 0.002]), np.ones(2))
        assert result == pytest.approx(10000.0 * 0.5)

    @patch(f"{_SABR_MOD}.metrics.rmsew")
    def test_params_dict_built_correctly(self, mock_rmsew):
        """LnVol/Beta/Nu/Rho must be taken from x[0..3] in order."""
        mock_rmsew.return_value = 0.0
        g = SabrGenerator(shift=0.0)
        captured = {}

        def capture(prices, targets, weights):
            captured['prices'] = prices
            return 0.0

        mock_rmsew.side_effect = capture
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.0]]))
        x = [0.11, 0.22, 0.33, -0.44]
        sabr_obj(x, g, 1.0, np.array([0.025]), 0.025, np.array([[0.002]]), np.ones(1))
        call_params = g.price_straddles_ref.call_args[0][3]
        assert call_params == {'LnVol': 0.11, 'Beta': 0.22, 'Nu': 0.33, 'Rho': -0.44}


class TestSabrGeneratorCalibrate:
    """Tests for calibrate (entirely uncovered)."""

    @patch(f"{_SABR_MOD}.opt.MultiOptimizer")
    def test_returns_correct_structure(self, mock_cls):
        mock_result = MagicMock()
        mock_result.x = [0.20, 0.5, 0.55, -0.25]
        mock_result.nfev = 100
        mock_cls.return_value.minimize.return_value = mock_result

        g = SabrGenerator(shift=0.0)
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.002, 0.002]]))

        expiries = np.array([0.5, 1.0])
        strikes = np.ones((2, 2)) * 0.025
        mkt_prices = np.ones((2, 2)) * 0.002
        cal_params, cal_prices, nfevs = g.calibrate(expiries, strikes, 0.025, mkt_prices,
                                                     weights=np.ones(2))

        assert len(cal_params) == 2
        assert set(cal_params[0].keys()) == {'LnVol', 'Beta', 'Nu', 'Rho'}
        assert nfevs == 200  # 2 expiries × 100 nfev each

    @patch(f"{_SABR_MOD}.opt.MultiOptimizer")
    def test_minimize_called_once_per_expiry(self, mock_cls):
        mock_result = MagicMock()
        mock_result.x = [0.20, 0.5, 0.55, -0.25]
        mock_cls.return_value.minimize.return_value = mock_result

        g = SabrGenerator(shift=0.0)
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.002]]))

        n_exp = 3
        expiries = np.array([0.5, 1.0, 2.0])
        strikes = np.ones((n_exp, 1)) * 0.025
        mkt_prices = np.ones((n_exp, 1)) * 0.002
        g.calibrate(expiries, strikes, 0.025, mkt_prices, weights=np.ones(1))

        assert mock_cls.return_value.minimize.call_count == n_exp

    @patch(f"{_SABR_MOD}.opt.MultiOptimizer")
    def test_cal_prices_length_matches_expiries(self, mock_cls):
        mock_result = MagicMock()
        mock_result.x = [0.20, 0.5, 0.55, -0.25]
        mock_cls.return_value.minimize.return_value = mock_result

        n_exp, n_str = 2, 3
        g = SabrGenerator(shift=0.0)
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.002] * n_str]))

        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        mkt_prices = np.ones((n_exp, n_str)) * 0.002
        _, cal_prices, _ = g.calibrate(expiries, strikes, 0.025, mkt_prices,
                                       weights=np.ones(n_str))

        assert len(cal_prices) == n_exp
        assert len(cal_prices[0]) == n_str

    @patch(f"{_SABR_MOD}.opt.MultiOptimizer")
    def test_cal_params_values_come_from_optimizer(self, mock_cls):
        """Calibrated parameters must be taken from result.x in LnVol/Beta/Nu/Rho order."""
        mock_result = MagicMock()
        mock_result.x = [0.11, 0.22, 0.33, -0.44]
        mock_cls.return_value.minimize.return_value = mock_result

        g = SabrGenerator(shift=0.0)
        g.price_straddles_ref = MagicMock(return_value=np.array([[0.002]]))

        expiries = np.array([1.0])
        strikes = np.ones((1, 1)) * 0.025
        mkt_prices = np.ones((1, 1)) * 0.002
        cal_params, _, _ = g.calibrate(expiries, strikes, 0.025, mkt_prices, weights=np.ones(1))

        assert cal_params[0] == {'LnVol': 0.11, 'Beta': 0.22, 'Nu': 0.33, 'Rho': -0.44}


class TestSabrGeneratorInit:
    def test_inherits_smile_generator(self):
        g = SabrGenerator(shift=0.03)
        assert g.shift == 0.03
        assert g.surface_size == 15 * 10


class TestSabrGeneratorPrice:
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_returns_correct_shape(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        g = SabrGenerator(shift=0.0)
        expiries = np.array([[0.5], [1.0]])
        strikes = np.array([[0.02, 0.03], [0.02, 0.03]])
        result = g.price(expiries, strikes, [[False, False], [False, False]], 0.025, SABR_PARAMS)
        assert result.shape == (2, 2)

    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_shift_applied_to_fwd(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        shift, fwd = 0.03, 0.01
        g = SabrGenerator(shift=shift)
        g.price(np.array([[1.0]]), np.array([[0.02]]), [[False]], fwd, SABR_PARAMS)
        # Third positional arg to sabr_from_dict is shifted_f
        assert mock_sabr.call_args[0][2] == pytest.approx(fwd + shift)


class TestSabrGeneratorPriceStraddlesRef:
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_returns_correct_shape(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        g = SabrGenerator(shift=0.0)
        expiries = np.array([0.5, 1.0])
        strikes = np.array([[0.02, 0.03], [0.02, 0.03]])
        result = g.price_straddles_ref(expiries, strikes, 0.025, SABR_PARAMS)
        assert result.shape == (2, 2)

    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_straddle_is_call_plus_put(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.002])
        g = SabrGenerator(shift=0.0)
        result = g.price_straddles_ref(np.array([1.0]), np.array([[0.025]]), 0.025, SABR_PARAMS)
        assert result[0][0] == pytest.approx(0.004)  # call[0] + put[0] = 0.002 + 0.002


class TestSabrGeneratorRetrieveDatasetsNoShuffle:
    def _make_df(self, n=5):
        return pd.DataFrame({
            'Ttm': np.ones(n), 'K': np.ones(n) * 0.02, 'F': np.ones(n) * 0.025,
            'LnVol': np.ones(n) * 0.2, 'Beta': np.ones(n) * 0.5,
            'Nu': np.ones(n) * 0.5, 'Rho': np.ones(n) * -0.25,
            'NVol': np.ones(n) * 0.005,
        })

    def test_x_set_has_7_columns(self):
        x, _ = SabrGenerator().retrieve_datasets_no_shuffle(self._make_df())
        assert x.shape[1] == 7

    def test_y_set_shape(self):
        _, y = SabrGenerator().retrieve_datasets_no_shuffle(self._make_df(n=5))
        assert y.shape == (5, 1)


class TestSabrGeneratorRetrieveInverseDatasetsNoShuffle:
    def _make_df(self, n=4, n_strikes=3):
        d = {'Ttm': np.ones(n), 'F': np.ones(n) * 0.025,
             'LnVol': np.ones(n) * 0.2, 'Beta': np.ones(n) * 0.5,
             'Nu': np.ones(n) * 0.5, 'Rho': np.ones(n) * -0.25}
        for i in range(n_strikes):
            d[f'K{i}'] = np.ones(n) * 0.001
        return pd.DataFrame(d)

    def test_x_set_has_2_plus_n_strikes_columns(self):
        x, _ = SabrGenerator().retrieve_inverse_datasets_no_shuffle(self._make_df(n_strikes=3))
        assert x.shape[1] == 5  # 2 base + 3 strikes

    def test_y_set_has_4_param_columns(self):
        _, y = SabrGenerator().retrieve_inverse_datasets_no_shuffle(self._make_df())
        assert y.shape[1] == 4


class TestSabrGeneratorPriceSurfaceMod:
    @patch(f"{_SABR_MOD}.bachelier.price")
    def test_output_shape(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 3, 4
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        expiries = np.array([0.5, 1.0, 2.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        result = SabrGenerator().price_surface_mod(
            model, expiries, strikes, [[False] * n_str] * n_exp, 0.025, SABR_PARAMS
        )
        assert result.shape == (n_exp, n_str)

    @patch(f"{_SABR_MOD}.bachelier.price")
    def test_model_receives_7_column_input(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 2, 3
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        SabrGenerator().price_surface_mod(
            model, expiries, strikes, [[False] * n_str] * n_exp, 0.025, SABR_PARAMS
        )
        assert model.predict.call_args[0][0].shape[1] == 7


class TestSabrGeneratorGenerateSamples:
    @patch(f"{_SABR_MOD}.black.price")
    @patch(f"{_SABR_MOD}.sabr.sabr_from_dict")
    def test_output_dataframe_columns(self, mock_sabr, mock_black):
        mock_sabr.return_value = 0.20
        mock_black.return_value = np.array([0.001])
        g = SabrGenerator(num_expiries=2, num_strikes=2, seed=0)
        rg = {'Ttm': [0.5, 1.0], 'K': [0.1, 0.9], 'F': [0.01, 0.04],
              'LnVol': [0.1, 0.3], 'Beta': [0.3, 0.7], 'Nu': [0.1, 0.5], 'Rho': [-0.5, 0.5]}
        df = g.generate_samples(g.surface_size, rg)
        assert set(df.columns) == {'Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price'}
