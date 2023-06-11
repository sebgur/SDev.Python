""" Select stovol generator based on type and other required information """
from volsurfacegen.sabrgenerator import SabrGenerator, ShiftedSabrGenerator
from volsurfacegen.mcsabrgenerator import McShiftedSabrGenerator
from volsurfacegen.fbsabrgenerator import FbSabrGenerator
from volsurfacegen.mczabrgenerator import McShiftedZabrGenerator
from volsurfacegen.mchestongenerator import McShiftedHestonGenerator


def set_generator(type_, num_expiries=10, surface_size=50, num_mc=100000,
                  points_per_year=25, seed=42):
    """ Select generator based on type and other information. Currently available are: SABR,
        ShiftedSABR, McShiftedSABR, FbSABR, McShiftedZABR and McShiftedHeston. All models
        except SABR and ShiftedSABR require inputs num_expiries and surface_size for sample
        generation, and num_mc and points_per_year for price calculation.
    """
    num_strikes = int(surface_size / num_expiries)

    if type_ == "SABR":
        generator = SabrGenerator(0.0, num_expiries, num_strikes, seed)
    elif type_ == "ShiftedSABR":
        generator = ShiftedSabrGenerator(num_expiries, num_strikes, seed)
    elif type_ == "McShiftedSABR":
        generator = McShiftedSabrGenerator(num_expiries, num_strikes, num_mc, points_per_year)
    elif type_ == "FbSABR":
        generator = FbSabrGenerator(num_expiries, num_strikes, num_mc, points_per_year)
    elif type_ == "McShiftedZABR":
        generator = McShiftedZabrGenerator(num_expiries, num_strikes, num_mc, points_per_year)
    elif type_ == "McShiftedHeston":
        generator = McShiftedHestonGenerator(num_expiries, num_strikes, num_mc, points_per_year)
    else:
        raise ValueError("Unknown model: " + type_)

    return generator

if __name__ == "__main__":
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 50
    NUM_EXPIRIES = 10
    TYPE = 'FbSABR'
    GENERATOR = set_generator(TYPE, NUM_EXPIRIES, SURFACE_SIZE, NUM_MC, POINTS_PER_YEAR)

    print("Setting generator model " + TYPE)
    print("Complete!")
