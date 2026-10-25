from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities.schedules import LegConvention, BondConvention


# US Treasuries: semi-annual ACT/ACT ICMA, SIFMA calendar, T+1, simple yield in the final coupon period
UST = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True),
                     settlement_lag=1, simple_final_period=True)

# UK gilts: semi-annual ACT/ACT ICMA, London calendar, T+1, ex-dividend 7 business days before each coupon
GILT = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False),
                      settlement_lag=1, ex_coupon_days=7)
