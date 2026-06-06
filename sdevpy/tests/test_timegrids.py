import numpy as np
import datetime as dt
from sdevpy.utilities import timegrids as tmg


def test_timegridbuilder_buckets():
    base = dt.datetime(2023, 1, 24)
    fixing = dt.datetime(2022, 1, 24)
    expiry = dt.datetime(2025, 1, 24)

    # Define buckets
    buckets = []
    buckets.append(tmg.TimeGridBucket(start=0.5, end=0.6, n_points=4))
    buckets.append(tmg.TimeGridBucket(start=1.0, end=1.1, n_points=4))

    builder = tmg.BucketTimeGridBuilder(buckets=buckets)
    builder.add_dates(base, [fixing, expiry])
    test = builder.complete_grid()
    # print(test)
    ref = np.asarray([0.5, 0.53333333, 0.56666667, 0.6, 1.0, 1.03333333, 1.06666667, 1.1, 2.00273973])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_timegridbuilder_dates():
    base = dt.datetime(2023, 1, 24)
    fixing = dt.datetime(2022, 1, 24)
    expiry = dt.datetime(2025, 1, 24)
    builder = tmg.SimpleTimeGridBuilder(points_per_year=3)
    builder.add_dates(base, [fixing, expiry])
    builder.refine()
    builder.clean()
    test = builder.get_grid()
    ref = np.asarray([0.40054795, 0.80109589, 1.20164384, 1.60219178, 2.00273973])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_timegridbuilder_times():
    time_grid_builder = tmg.SimpleTimeGridBuilder(points_per_year=3)
    expiries = np.asarray([1.4, 0.125, 0.5]).reshape(-1, 1)
    time_grid_builder.add_grid(expiries)
    test = time_grid_builder.complete_grid()
    ref = np.asarray([0.125, 0.46666667, 0.5, 0.93333333, 1.4])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_model_time():
    base = dt.datetime(2023, 1, 24)
    fixing = dt.datetime(2022, 1, 24)
    expiry = dt.datetime(2025, 1, 24)

    test = []
    test.append(tmg.model_time(base, fixing))
    test.extend(tmg.model_time(base, [fixing, expiry]))
    test = np.asarray(test)

    ref = np.asarray([-1.0, -1.0, 2.0027397260273974])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)

################ build_sparse_timegrid ############################################################

def test_build_sparse_timegrid_short_term():
    # t=0.15 is before first granularity point
    test = tmg.build_sparse_timegrid(0.15)
    ref = np.asarray([0.15])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_build_sparse_timegrid_mid_term():
    # t=0.3 is far enough from 0.25 that a new point is added
    test = tmg.build_sparse_timegrid(0.3)
    ref = np.asarray([0.25, 0.3])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_build_sparse_timegrid_term_tol():
    # t=0.26 is within term_tol of 0.25, so 0.25 is replaced
    test = tmg.build_sparse_timegrid(0.26)
    ref = np.asarray([0.26])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_build_sparse_timegrid_exact_point():
    test = tmg.build_sparse_timegrid(1.0)
    ref = np.asarray([0.25, 0.5, 0.75, 1.0])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_build_sparse_timegrid_long_term():
    test = tmg.build_sparse_timegrid(5.0)
    ref = np.asarray([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    test_timegridbuilder_buckets()
    # test_timegridbuilder_dates()
    # test_timegridbuilder_times()
    # test_model_time()
