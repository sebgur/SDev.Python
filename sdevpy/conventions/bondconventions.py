from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities.schedules import LegConvention, BondConvention, YieldType, BillConvention


################## BONDS ##############################################################################################
# US Treasuries: semi-annual ACT/ACT ICMA, SIFMA calendar, T+1, simple yield in the final coupon period
UST = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True),
                     settlement_lag=1, simple_final_period=True)

# UK gilts: semi-annual ACT/ACT ICMA, London calendar, T+1, ex-dividend 7 business days before each coupon
GILT = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False),
                      settlement_lag=1, ex_coupon_days=7)

# JGBs: semi-annual on the 20th, Tokyo market calendar, T+1, accrued ACT/365F capped at c/2 (JNX), simple yield.
# A full first coupon is a term of each bond: build it with accrual_start=previous_coupon_date(issue, ...)
JGB = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'JPGOV', bdc=BDC.F, adjust_accrual=False),
                     settlement_lag=1, accrued_daycount=DayCount.ACT365F, accrued_cap=True,
                     stub_daycount=DayCount.ACT365F, stub_complement=True, yield_type=YieldType.SIMPLE)

# German Bunds/Bobls/Schatz: annual ACT/ACT ICMA, TARGET, T+2, street yield
BUND = BondConvention(LegConvention('1Y', DayCount.ACTACT_ICMA, 'EUR', bdc=BDC.F, adjust_accrual=False),
                      settlement_lag=2)

# Italian BTPs: semi-annual ACT/ACT ICMA, TARGET, T+2. Yield quoted annual-compounded:
BTP = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'EUR', bdc=BDC.F, adjust_accrual=False),
                     settlement_lag=2, yield_frequency=1)

# Government of Canada: semi-annual, Canadian bond market calendar, T+1. Canadian accrued and odd coupons
# Act/365 (Canadian bond), money-market yield in the final coupon period (IIAC)
CAN = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'CAGOV', bdc=BDC.F, adjust_accrual=False),
                     settlement_lag=1, accrued_daycount=DayCount.ACT365_CANADIAN, simple_final_period=True,
                     final_period_daycount=DayCount.ACT365F, stub_daycount=DayCount.ACT365_CANADIAN)

# Australian Commonwealth Government Bonds: semi-annual ACT/ACT ICMA, Sydney, T+2. Ex-interest 7 calendar days
# before each coupon
ACGB = BondConvention(LegConvention('6M', DayCount.ACTACT_ICMA, 'AUGOV', bdc=BDC.F, adjust_accrual=False),
                      settlement_lag=2, ex_coupon_days=7, ex_coupon_calendar_days=True)


################## BILLS ##############################################################################################
UST_BILL = BillConvention(YieldType.STREET)  # bond-equivalent yield; also quoted on discount rate (ACT/360)
JGB_BILL = BillConvention(YieldType.SIMPLE)  # T-Bills: ACT/365F under a year (simple_yield_time)
BUBILL = BillConvention(YieldType.SIMPLE, DayCount.ACT360)
BOT = BillConvention(YieldType.SIMPLE, DayCount.ACT360)
CAN_BILL = BillConvention(YieldType.SIMPLE, DayCount.ACT365F)
ACGB_NOTE = BillConvention(YieldType.SIMPLE, DayCount.ACT365F)  # Treasury Notes
