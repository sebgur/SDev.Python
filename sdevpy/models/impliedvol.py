from abc import ABC, abstractmethod


class Section(ABC):
    @abstractmethod
    def value(self, t, x):
        pass

    @abstractmethod
    def dump(self):
        pass


class ParamSection(Section):
    def __init__(self, time, formula):
        self.params = None
        self.formula = formula
        self.model = None
        self.time = time

    def value(self, t, x):
        """ In the base, we simply use the formula on the parameters. In inherited classes, we
            may have a more complex behaviour such as applying the formula to a transformed
            set of parameters. """
        return self.formula(t, x, self.params)

    def update_params(self, new_params):
        """ In the base, we only copy the new parameters in. Inherited classes may do more. """
        self.params = new_params.copy()

    def check_params(self):
        """ In the base, all parameters are allowed so we always answer True and penalty = 0.0.
            Inherited classes may have constraints and calculate non-trivial penalties. """
        return True, 0.0

    @abstractmethod
    def constraints(self):
        pass

    @abstractmethod
    def dump_params(self):
        pass

    def dump(self):
        data = {'time': self.time, 'model': self.model, 'params': self.dump_params()}
        return data
