""" Toy examples for understanding functools.lru_cache """
import time
from functools import lru_cache


# --- 1. Basic memoization: expensive call becomes free on repeat ---------

@lru_cache(maxsize=None)
def slow_square(x):
    time.sleep(0.5)  # pretend this is expensive (like building a calendar)
    return x * x


def demo_basic():
    print("\n--- demo_basic ---")
    t0 = time.time()
    print(slow_square(4))          # miss -> ~0.5s
    print(f"first call: {time.time() - t0:.2f}s")

    t0 = time.time()
    print(slow_square(4))          # hit -> instant
    print(f"second call (cached): {time.time() - t0:.3f}s")

    print(slow_square.cache_info())  # hits=1, misses=1, currsize=1


# --- 2. maxsize enforces real LRU eviction --------------------------------

@lru_cache(maxsize=2)
def tag(x):
    print(f"  computing tag({x})")
    return f"tag-{x}"


def demo_eviction():
    print("\n--- demo_eviction ---")
    tag(1)              # miss, cache: [1]
    tag(2)              # miss, cache: [1,2]
    tag(1)              # hit,  cache: [2,1]  (1 becomes most-recently-used)
    tag(3)              # miss, evicts 2 (least recently used), cache: [1,3]
    tag(2)              # miss again -- 2 was evicted
    print(tag.cache_info())


# --- 3. Gotcha: unhashable arguments raise TypeError ----------------------

@lru_cache(maxsize=None)
def process(data):
    return sum(data)


def demo_unhashable():
    print("\n--- demo_unhashable ---")
    try:
        process([1, 2, 3])   # list is unhashable
    except TypeError as e:
        print(f"TypeError as expected: {e}")
    print(process((1, 2, 3)))  # tuple works fine


# --- 4. Gotcha: cached return value is the SAME object -- mutation bug ----

@lru_cache(maxsize=None)
def get_market_data(pricing_date, ccy_pair):
    print(f"  building dataset for {pricing_date}, {ccy_pair}")
    return {"spot": 1.10, "vols": [0.1, 0.12, 0.15]}


def demo_mutation_bug():
    print("\n--- demo_mutation_bug ---")
    ds1 = get_market_data("2026-01-01", "EURUSD")
    ds1["spot"] = 999.0          # caller mutates the dict in place!
    ds2 = get_market_data("2026-01-01", "EURUSD")  # same cache entry
    print(f"ds2['spot'] = {ds2['spot']}")  # prints 999.0, not 1.10 -- corrupted!


if __name__ == "__main__":
    demo_basic()
    demo_eviction()
    demo_unhashable()
    demo_mutation_bug()
