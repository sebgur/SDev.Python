

class Cashflow:
    def __init__(self, index, paydate, notional=1.0, accrual=1.0):
        self.index = index
        self.paydate = paydate
        self.notional = notional
        self.accrual = accrual

    def calculate(self, market_state):
        index_amount = self.index.evaluate(market_state)
        return index_amount * self.notional * self.accrual
        # schedule = CashflowSchedule()
        # schedule.add_cashflow(self.payment_time, amount)
        # return schedule


# class CashflowLeg:
#     def __init__(self):
#         self.cashflows = {}

#     def add_cashflow(self, time, amount):
#         if time in self.cashflows:
#             self.cashflows[time] += amount
#         else:
#             self.cashflows[time] = amount.copy()

#     def __add__(self, other):
#         result = CashflowSchedule()
#         for t, a in self.cashflows.items():
#             result.add_cashflow(t, a)

#         for t, a in other.cashflows.items():
#             result.add_cashflow(t, a)

#         return result


class DiscountEngine:
    def __init__(self, curve_times, discount_factors):
        self.curve_times = curve_times
        self.discount_factors = discount_factors

    def df(self, t):
        return np.interp(t, self.curve_times, self.discount_factors)


if __name__ == "__main__":
    print("Hello")
