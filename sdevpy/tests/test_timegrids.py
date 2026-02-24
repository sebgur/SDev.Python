import numpy as np
import datetime as dt
from sdevpy.tools import timegrids as tmg


def test_timegridbuilder_dates():
    base = dt.datetime(2023, 1, 24)
    fixing = dt.datetime(2022, 1, 24)
    expiry = dt.datetime(2025, 1, 24)
    builder = tmg.SimpleTimeGridBuilder(3)
    builder.add_dates(base, [fixing, expiry])
    builder.refine()
    builder.clean()
    test = np.asarray(builder.time_grid_)
    ref = np.asarray([0.40054795, 0.80109589, 1.20164384, 1.60219178, 2.00273973])
    assert np.allclose(test, ref, 1e-10)


def test_timegridbuilder_times():
    time_grid_builder = tmg.SimpleTimeGridBuilder(points_per_year=3)
    EXPIRIES = np.asarray([1.4, 0.125, 0.5]).reshape(-1, 1)
    time_grid_builder.add_grid(EXPIRIES)
    time_grid_builder.refine()
    time_grid_builder.clean()
    test = np.asarray(time_grid_builder.time_grid_)
    print(test)
    ref = np.asarray([0.125, 0.46666667, 0.5, 0.93333333, 1.4])
    assert np.allclose(test, ref, 1e-10)


def test_model_time():
    base = dt.datetime(2023, 1, 24)
    fixing = dt.datetime(2022, 1, 24)
    expiry = dt.datetime(2025, 1, 24)

    test = []
    test.append(tmg.model_time(base, fixing))
    test.extend(tmg.model_time(base, [fixing, expiry]))
    test = np.asarray(test)
    print(test)

    ref = np.asarray([-1.0, -1.0, 2.0027397260273974])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    test_timegridbuilder_dates()
    test_timegridbuilder_times()
    test_model_time()
