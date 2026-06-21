import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from sdevpy.volatility.impliedvol.models import fbsabr
from sdevpy.volatility.impliedvol.models import mcheston
from sdevpy.volatility.impliedvol.models import mcsabr
from sdevpy.volatility.impliedvol.models import mczabr
from sdevpy.volatility.mlsurfacegen.smilegenerator import SmileGenerator
from sdevpy.volatility.mlsurfacegen.sabrgenerator import SabrGenerator
from sdevpy.volatility.mlsurfacegen.mcsabrgenerator import McSabrGenerator
from sdevpy.volatility.mlsurfacegen.fbsabrgenerator import FbSabrGenerator
from sdevpy.volatility.mlsurfacegen.mczabrgenerator import McZabrGenerator
from sdevpy.volatility.mlsurfacegen.mchestongenerator import McHestonGenerator
from sdevpy.volatility.mlsurfacegen import stovolfactory


# ── SmileGenerator ───────────────────────────────────────────────────────────

class ConcreteSmileGen(SmileGenerator):
    """Minimal concrete subclass for testing SmileGenerator non-abstract methods."""
    def generate_samples(self, num_samples, rg): pass
    def generate_samples_inverse(self, num_samples, rg, spreads): pass
    def price(self, expiries, strikes, are_calls, fwd, parameters): return MagicMock()
    def price_straddles_ref(self, expiries, strikes, fwd, parameters): pass
    def retrieve_datasets_no_shuffle(self, data_df): pass
    def retrieve_inverse_datasets_no_shuffle(self, data_df): pass
    def price_surface_mod(self, model, expiries, strikes, are_calls, fwd, parameters): pass


class TestSmileGeneratorInit:
    def test_surface_size_is_expiries_times_strikes(self):
        g = ConcreteSmileGen(num_expiries=5, num_strikes=3)
        assert g.surface_size == 15

    def test_defaults(self):
        g = ConcreteSmileGen()
        assert g.shift == 0.0
        assert g.num_expiries == 15
        assert g.num_strikes == 10

    def test_is_call_false_by_default(self):
        g = ConcreteSmileGen()
        assert g.is_call is False
        assert g.target_is_call() is False

    def test_are_calls_shape(self):
        g = ConcreteSmileGen(num_expiries=3, num_strikes=4)
        assert len(g.are_calls) == 3
        assert len(g.are_calls[0]) == 4


class TestSmileGeneratorConvertStrikes:
    def setup_method(self):
        self.g = ConcreteSmileGen(shift=0.0)

    def test_strikes_passthrough(self):
        strikes = np.array([[0.01, 0.02], [0.03, 0.04]])
        result = self.g.convert_strikes(None, strikes, 0.02, {}, 'Strikes')
        np.testing.assert_array_equal(result, strikes)

    def test_spreads_converts_to_absolute(self):
        fwd = 0.02
        spreads = np.array([[100.0, -100.0]])
        result = self.g.convert_strikes(None, spreads, fwd, {}, 'Spreads')
        np.testing.assert_array_almost_equal(result, fwd + spreads / 10000.0)

    def test_percentiles_at_50th(self):
        g = ConcreteSmileGen(shift=0.0)
        expiries = np.array([[1.0]])
        fwd, lnvol = 0.03, 0.2
        # norm.ppf(0.5) = 0 → strike = fwd * exp(-0.5*stdev^2)
        result = g.convert_strikes(expiries, np.array([[0.5]]), fwd, {'LnVol': lnvol}, 'Percentiles')
        stdev = lnvol * np.sqrt(1.0)
        expected = fwd * np.exp(-0.5 * stdev**2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_percentiles_with_shift(self):
        shift = 0.03
        g = ConcreteSmileGen(shift=shift)
        expiries = np.array([[1.0]])
        fwd, lnvol = 0.01, 0.2
        result = g.convert_strikes(expiries, np.array([[0.5]]), fwd, {'LnVol': lnvol}, 'Percentiles')
        sfwd = fwd + shift
        stdev = lnvol * np.sqrt(1.0)
        expected = sfwd * np.exp(-0.5 * stdev**2) - shift
        np.testing.assert_array_almost_equal(result, expected)

    def test_percentiles_raises_without_lnvol(self):
        with pytest.raises(RuntimeError, match="Lognormal vol"):
            self.g.convert_strikes(np.array([[1.0]]), np.array([[0.5]]), 0.02, {}, 'Percentiles')

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid strike input method"):
            self.g.convert_strikes(None, None, 0.02, {}, 'BadMethod')


class TestSmileGeneratorPriceSurfaceRef:
    def test_delegates_to_price(self):
        g = ConcreteSmileGen()
        g.price = MagicMock(return_value=np.array([1.0]))
        result = g.price_surface_ref("exp", "str", "calls", 0.02, {})
        g.price.assert_called_once_with("exp", "str", "calls", 0.02, {})
        np.testing.assert_array_equal(result, np.array([1.0]))


class TestSmileGeneratorFromToFile:
    def test_roundtrip(self, tmp_path):
        df = pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
        path = tmp_path / "data.tsv"
        SmileGenerator.to_file(df, path)
        loaded = SmileGenerator.from_file(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_from_file_shuffle_preserves_values(self, tmp_path):
        df = pd.DataFrame({'A': np.arange(100, dtype=float)})
        path = tmp_path / "data.tsv"
        SmileGenerator.to_file(df, path)
        loaded = SmileGenerator.from_file(path, shuffle=True)
        assert set(loaded['A']) == set(df['A'])
        assert loaded['A'].tolist() != df['A'].tolist()  # order should differ


class TestSmileGeneratorRetrieveDatasets:
    def test_retrieve_datasets_from_df_calls_no_shuffle(self):
        g = ConcreteSmileGen()
        df = pd.DataFrame({'X': [1, 2]})
        g.retrieve_datasets_no_shuffle = MagicMock(return_value=("x", "y"))
        result = g.retrieve_datasets_from_df(df, shuffle=False)
        g.retrieve_datasets_no_shuffle.assert_called_once_with(df)
        assert result == ("x", "y")

    def test_retrieve_inverse_datasets_from_df_calls_no_shuffle(self):
        g = ConcreteSmileGen()
        df = pd.DataFrame({'X': [1, 2]})
        g.retrieve_inverse_datasets_no_shuffle = MagicMock(return_value=("x", "y"))
        result = g.retrieve_inverse_datasets_from_df(df, shuffle=False)
        g.retrieve_inverse_datasets_no_shuffle.assert_called_once_with(df)
        assert result == ("x", "y")


# ── SabrGenerator ────────────────────────────────────────────────────────────

SABR_PARAMS = {'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25}
_SABR_MOD = "sdevpy.volatility.mlsurfacegen.sabrgenerator"


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


# ── McSabrGenerator ──────────────────────────────────────────────────────────

_MCSABR_MOD = "sdevpy.volatility.mlsurfacegen.mcsabrgenerator"


class TestMcSabrGeneratorInit:
    def test_stores_mc_params(self):
        g = McSabrGenerator(shift=0.03, num_mc=5000, points_per_year=20)
        assert g.num_mc == 5000
        assert g.points_per_year == 20
        assert g.shift == 0.03


class TestMcSabrGeneratorPrice:
    @patch(f"{_MCSABR_MOD}.mcsabr.price")
    def test_delegates_to_mcsabr(self, mock_price):
        mock_price.return_value = np.ones((2, 2))
        g = McSabrGenerator(shift=0.03, num_mc=100, points_per_year=10)
        result = g.price(
            np.array([[0.5], [1.0]]), np.ones((2, 2)) * 0.02,
            [[False, False], [False, False]], 0.01, SABR_PARAMS
        )
        mock_price.assert_called_once()
        np.testing.assert_array_equal(result, np.ones((2, 2)))

    @patch(f"{_MCSABR_MOD}.mcsabr.price")
    def test_shift_applied_to_strikes_and_fwd(self, mock_price):
        mock_price.return_value = np.ones((1, 1))
        shift, fwd = 0.03, 0.01
        strike = np.array([[0.02]])
        g = McSabrGenerator(shift=shift, num_mc=100, points_per_year=10)
        g.price(np.array([[1.0]]), strike, [[False]], fwd, SABR_PARAMS)
        args = mock_price.call_args[0]
        np.testing.assert_array_almost_equal(args[1], strike + shift)  # shifted_k
        assert args[3] == pytest.approx(fwd + shift)                   # shifted_f


# ── FbSabrGenerator ──────────────────────────────────────────────────────────

_FBSABR_MOD = "sdevpy.volatility.mlsurfacegen.fbsabrgenerator"


class TestFbSabrGeneratorInit:
    def test_shift_hardcoded_to_003(self):
        g = FbSabrGenerator(num_mc=100, points_per_year=10)
        assert g.shift == pytest.approx(0.03)

    def test_stores_mc_params(self):
        g = FbSabrGenerator(num_mc=2000, points_per_year=15)
        assert g.num_mc == 2000
        assert g.points_per_year == 15


class TestFbSabrGeneratorPrice:
    @patch(f"{_FBSABR_MOD}.fbsabr.price")
    def test_delegates_to_fbsabr(self, mock_price):
        mock_price.return_value = np.ones((1, 2))
        g = FbSabrGenerator(num_mc=100, points_per_year=10)
        result = g.price(np.array([[1.0]]), np.array([[0.01, 0.02]]), [[False, False]], 0.01, SABR_PARAMS)
        mock_price.assert_called_once()
        np.testing.assert_array_equal(result, np.ones((1, 2)))

    @patch(f"{_FBSABR_MOD}.fbsabr.price")
    def test_passes_num_mc_and_points_per_year(self, mock_price):
        mock_price.return_value = np.ones((1, 1))
        g = FbSabrGenerator(num_mc=500, points_per_year=25)
        g.price(np.array([[1.0]]), np.array([[0.02]]), [[False]], 0.01, SABR_PARAMS)
        args = mock_price.call_args[0]
        assert args[5] == 500
        assert args[6] == 25


# ── McZabrGenerator ──────────────────────────────────────────────────────────

_MCZABR_MOD = "sdevpy.volatility.mlsurfacegen.mczabrgenerator"
ZABR_PARAMS = {'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25, 'Gamma': 0.7}


class TestMcZabrGeneratorInit:
    def test_stores_mc_params(self):
        g = McZabrGenerator(shift=0.02, num_mc=3000, points_per_year=12)
        assert g.num_mc == 3000
        assert g.points_per_year == 12
        assert g.shift == 0.02


class TestMcZabrGeneratorPrice:
    @patch(f"{_MCZABR_MOD}.mczabr.price")
    def test_delegates_to_mczabr(self, mock_price):
        mock_price.return_value = np.ones((2, 2))
        g = McZabrGenerator(shift=0.03, num_mc=100, points_per_year=10)
        result = g.price(
            np.array([[1.0], [2.0]]), np.ones((2, 2)) * 0.02,
            [[False, False], [False, False]], 0.01, ZABR_PARAMS
        )
        mock_price.assert_called_once()
        np.testing.assert_array_equal(result, np.ones((2, 2)))

    @patch(f"{_MCZABR_MOD}.mczabr.price")
    def test_shift_applied_to_fwd(self, mock_price):
        mock_price.return_value = np.ones((1, 1))
        shift, fwd = 0.03, 0.01
        g = McZabrGenerator(shift=shift, num_mc=100, points_per_year=10)
        g.price(np.array([[1.0]]), np.array([[0.02]]), [[False]], fwd, ZABR_PARAMS)
        assert mock_price.call_args[0][3] == pytest.approx(fwd + shift)


class TestMcZabrGeneratorRetrieveDatasetsNoShuffle:
    def _make_df(self, n=5):
        return pd.DataFrame({
            'Ttm': np.ones(n), 'K': np.ones(n) * 0.02, 'F': np.ones(n) * 0.025,
            'LnVol': np.ones(n) * 0.2, 'Beta': np.ones(n) * 0.5, 'Nu': np.ones(n) * 0.5,
            'Rho': np.ones(n) * -0.25, 'Gamma': np.ones(n) * 0.7, 'NVol': np.ones(n) * 0.005,
        })

    def test_x_set_has_8_columns(self):
        x, _ = McZabrGenerator().retrieve_datasets_no_shuffle(self._make_df())
        assert x.shape[1] == 8

    def test_y_set_shape(self):
        _, y = McZabrGenerator().retrieve_datasets_no_shuffle(self._make_df(n=7))
        assert y.shape == (7, 1)


class TestMcZabrGeneratorPriceSurfaceMod:
    @patch(f"{_MCZABR_MOD}.bachelier.price")
    def test_output_shape(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 2, 3
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        result = McZabrGenerator().price_surface_mod(
            model, expiries, strikes, [[False] * n_str] * n_exp, 0.025, ZABR_PARAMS
        )
        assert result.shape == (n_exp, n_str)

    @patch(f"{_MCZABR_MOD}.bachelier.price")
    def test_model_receives_8_column_input(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 2, 3
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        McZabrGenerator().price_surface_mod(
            model, np.array([0.5, 1.0]), np.ones((n_exp, n_str)) * 0.025,
            [[False] * n_str] * n_exp, 0.025, ZABR_PARAMS
        )
        assert model.predict.call_args[0][0].shape[1] == 8


# ── McHestonGenerator ────────────────────────────────────────────────────────

_MCHESTON_MOD = "sdevpy.volatility.mlsurfacegen.mchestongenerator"
HESTON_PARAMS = {'LnVol': 0.20, 'Kappa': 1.0, 'Theta': 0.04, 'Xi': 0.30, 'Rho': -0.25}


class TestMcHestonGeneratorInit:
    def test_stores_mc_params(self):
        g = McHestonGenerator(shift=0.02, num_mc=4000, points_per_year=15)
        assert g.num_mc == 4000
        assert g.points_per_year == 15
        assert g.shift == 0.02


class TestMcHestonGeneratorPrice:
    @patch(f"{_MCHESTON_MOD}.mcheston.price")
    def test_delegates_to_mcheston(self, mock_price):
        mock_price.return_value = np.ones((2, 2))
        g = McHestonGenerator(shift=0.03, num_mc=100, points_per_year=10)
        result = g.price(
            np.array([[1.0], [2.0]]), np.ones((2, 2)) * 0.02,
            [[False, False], [False, False]], 0.01, HESTON_PARAMS
        )
        mock_price.assert_called_once()
        np.testing.assert_array_equal(result, np.ones((2, 2)))

    @patch(f"{_MCHESTON_MOD}.mcheston.price")
    def test_shift_applied_to_fwd(self, mock_price):
        mock_price.return_value = np.ones((1, 1))
        shift, fwd = 0.03, 0.01
        g = McHestonGenerator(shift=shift, num_mc=100, points_per_year=10)
        g.price(np.array([[1.0]]), np.array([[0.02]]), [[False]], fwd, HESTON_PARAMS)
        assert mock_price.call_args[0][3] == pytest.approx(fwd + shift)


class TestMcHestonGeneratorRetrieveDatasetsNoShuffle:
    def _make_df(self, n=5):
        return pd.DataFrame({
            'Ttm': np.ones(n), 'K': np.ones(n) * 0.02, 'F': np.ones(n) * 0.025,
            'LnVol': np.ones(n) * 0.2, 'Kappa': np.ones(n) * 1.0, 'Theta': np.ones(n) * 0.04,
            'Xi': np.ones(n) * 0.3, 'Rho': np.ones(n) * -0.25, 'NVol': np.ones(n) * 0.005,
        })

    def test_x_set_has_8_columns(self):
        x, _ = McHestonGenerator().retrieve_datasets_no_shuffle(self._make_df())
        assert x.shape[1] == 8

    def test_y_set_shape(self):
        _, y = McHestonGenerator().retrieve_datasets_no_shuffle(self._make_df(n=6))
        assert y.shape == (6, 1)


class TestMcHestonGeneratorPriceSurfaceMod:
    @patch(f"{_MCHESTON_MOD}.bachelier.price")
    def test_output_shape(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 2, 3
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        expiries = np.array([0.5, 1.0])
        strikes = np.ones((n_exp, n_str)) * 0.025
        result = McHestonGenerator().price_surface_mod(
            model, expiries, strikes, [[False] * n_str] * n_exp, 0.025, HESTON_PARAMS
        )
        assert result.shape == (n_exp, n_str)

    @patch(f"{_MCHESTON_MOD}.bachelier.price")
    def test_model_receives_8_column_input(self, mock_bach):
        mock_bach.return_value = 0.001
        model = MagicMock()
        n_exp, n_str = 2, 3
        model.predict.return_value = np.ones(n_exp * n_str) * 0.005
        McHestonGenerator().price_surface_mod(
            model, np.array([0.5, 1.0]), np.ones((n_exp, n_str)) * 0.025,
            [[False] * n_str] * n_exp, 0.025, HESTON_PARAMS
        )
        assert model.predict.call_args[0][0].shape[1] == 8


# ── stovolfactory ────────────────────────────────────────────────────────────

class TestStovolFactory:
    def test_sabr_returns_sabr_generator(self):
        assert isinstance(stovolfactory.set_generator("SABR"), SabrGenerator)

    def test_mcsabr_returns_mcsabr_generator(self):
        assert isinstance(stovolfactory.set_generator("McSABR"), McSabrGenerator)

    def test_fbsabr_returns_fbsabr_generator(self):
        assert isinstance(stovolfactory.set_generator("FbSABR"), FbSabrGenerator)

    def test_mczabr_returns_mczabr_generator(self):
        assert isinstance(stovolfactory.set_generator("McZABR"), McZabrGenerator)

    def test_mcheston_returns_mcheston_generator(self):
        assert isinstance(stovolfactory.set_generator("McHeston"), McHestonGenerator)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown model"):
            stovolfactory.set_generator("UnknownModel")

    def test_params_forwarded_to_sabr(self):
        g = stovolfactory.set_generator("SABR", shift=0.02, num_expiries=5, num_strikes=3)
        assert g.shift == pytest.approx(0.02)
        assert g.num_expiries == 5
        assert g.num_strikes == 3

    def test_mc_params_forwarded(self):
        g = stovolfactory.set_generator("McSABR", num_mc=500, points_per_year=20)
        assert g.num_mc == 500
        assert g.points_per_year == 20


######################## PREVIOUS ##############
FWD = 0.05

############# MC FBSABR ###########################################################################
FBSABR_PARAMS = {'LnVol': 0.30, 'Beta': 0.5, 'Nu': 0.50, 'Rho': 0.0}


def test_mcfbsabr_calculate_alpha():
    result = fbsabr.calculate_fbsabr_alpha(0.30, 0.05, 0.5)
    expected = 0.30 * (0.05 ** 0.5)
    assert np.isclose(result, expected)


def test_mcfbsabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = fbsabr.price(expiries, strikes, are_calls, FWD, FBSABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcfbsabr_euler_scheme():
    expiries = np.asarray([1.0])
    strikes = [[FWD]]
    are_calls = [[True]]
    result = fbsabr.price(expiries, strikes, are_calls, FWD, FBSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='Euler')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


############# MC HESTON ###########################################################################
HESTON_PARAMS = {'LnVol': 0.25, 'Kappa': 1.0, 'Theta': 0.0625, 'Xi': 0.50, 'Rho': -0.25}


def test_mcheston_calculate_v0():
    assert np.isclose(mcheston.calculate_v0(0.25), 0.0625)


def test_mcheston_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mcheston.price(expiries, strikes, are_calls, FWD, HESTON_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcheston_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mcheston.price(expiries, strikes, are_calls, FWD, HESTON_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


############# MC SABR ###########################################################################
MCSABR_PARAMS = {'LnVol': 0.25, 'Beta': 0.5, 'Nu': 0.50, 'Rho': -0.25}


def test_mcsabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mcsabr.price(expiries, strikes, are_calls, FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcsabr_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mcsabr.price(expiries, strikes, are_calls, FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


def test_mcsabr_log_euler_scheme():
    expiries = np.asarray([1.0])
    result = mcsabr.price(expiries, [[FWD]], [[True]], FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='LogEuler')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


def test_mcsabr_andersen_scheme():
    expiries = np.asarray([1.0])
    result = mcsabr.price(expiries, [[FWD]], [[True]], FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='Andersen')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


############# MC ZABR ###########################################################################
MCZABR_PARAMS = {'LnVol': 0.25, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.0}


def test_mczabr_calculate_alpha():
    result = mczabr.calculate_zabr_alpha(0.25, 0.05, 0.7)
    expected = 0.25 * (0.05 ** 0.3)
    assert np.isclose(result, expected)
    # beta=1: alpha = ln_vol * fwd^0 = ln_vol
    assert np.isclose(mczabr.calculate_zabr_alpha(0.25, 0.05, 1.0), 0.25)


def test_mczabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mczabr_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


def test_mczabr_price_gamma_one_close_to_mcsabr():
    # ZABR with gamma=1 reduces to SABR (LogEuler scheme) — prices should be close
    sabr_params = {k: v for k, v in MCZABR_PARAMS.items() if k != 'Gamma'}
    expiries = np.asarray([1.0])
    strikes = [[FWD]]
    are_calls = [[True]]
    z = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=2000, points_per_year=10)
    s = mcsabr.price(expiries, strikes, are_calls, FWD, sabr_params, num_mc=2000, points_per_year=10,
                   scheme='LogEuler')
    assert np.abs(z[0, 0] - s[0, 0]) < 5e-4  # within 0.5bp forward


if __name__ == "__main__":
    test = TestMcZabrGeneratorInit()
    test.test_stores_mc_params()
