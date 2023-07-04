""" Select stovol generator based on type and other required information """
from sdevpy.volsurfacegen.sabrgenerator import SabrGenerator
from sdevpy.volsurfacegen.mcsabrgenerator import McSabrGenerator
from sdevpy.volsurfacegen.fbsabrgenerator import FbSabrGenerator
from sdevpy.volsurfacegen.mczabrgenerator import McZabrGenerator
from sdevpy.volsurfacegen.mchestongenerator import McHestonGenerator


def set_generator(type_, shift=0.03, num_expiries=10, num_strikes=5, num_mc=100000,
                  points_per_year=25, seed=42):
    """ Select generator based on type and other information. Currently available are: SABR,
        McSABR, FbSABR, McZABR and McHeston. """
    if type_ == "SABR":
        generator = SabrGenerator(shift, num_expiries, num_strikes, seed)
    elif type_ == "McSABR":
        generator = McSabrGenerator(shift, num_expiries, num_strikes, num_mc, points_per_year, seed)
    elif type_ == "FbSABR":
        generator = FbSabrGenerator(num_expiries, num_strikes, num_mc, points_per_year, seed)
    elif type_ == "McZABR":
        generator = McZabrGenerator(shift, num_expiries, num_strikes, num_mc, points_per_year, seed)
    elif type_ == "McHeston":
        generator = McHestonGenerator(shift, num_expiries, num_strikes, num_mc, points_per_year, seed)
    else:
        raise ValueError("Unknown model: " + type_)

    return generator

if __name__ == "__main__":
    NUM_MC = 1 * 1000
    POINTS_PER_YEAR = 25
    NUM_EXPIRIES = 10
    NUM_STRIKES = 5
    TYPE = 'FbSABR'
    SHIFT = 0.03
    GENERATOR = set_generator(TYPE, SHIFT, NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

    print("Setting generator model " + TYPE)
    print("Complete!")
