

# class Cashflow:
#     def __init__(self, time, amount):
#         self.time = time  # scalar
#         self.amount = amount  # vector (n_paths,)


class CashflowSchedule:
    def __init__(self):
        self.dates = []
        self.amounts = []

    def add(self, date, amount):
        self.dates.append(date)
        self.amounts.append(amount)

    def aggregate(self):
        return np.array(self.times), np.column_stack(self.amounts)


class DiscountEngine:
    def __init__(self, curve_times, discount_factors):
        self.curve_times = curve_times
        self.discount_factors = discount_factors

    def df(self, t):
        return np.interp(t, self.curve_times, self.discount_factors)


if __name__ == "__main__":
    print("Hello")
