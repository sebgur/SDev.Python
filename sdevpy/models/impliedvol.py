from abc import ABC, abstractmethod


class Section(ABC):
    @abstractmethod
    def value(self, t, x):
        pass


class ParamSection(Section):
    def __init__(self, formula):
        self.params = None
        self.formula = formula

    def value(self, t, x):
        return self.formula(t, x, self.params)

    def update_params(self, new_params):
        """ In the base, we only copy the new parameters in. Inherited classes
            may do more. """
        self.params = new_params.copy()

    def check_params(self):
        """ In the base, all parameters are allowed so we always answer True
            and penalty = 0.0. Inherited classes may have constraint and calculate
            non-trivial penalties. """
        return True, 0.0
