from abc import ABC, abstractmethod


class Product(ABC):
    @abstractmethod
    def disc_payoff(self, paths, disc_curve):
        pass

    @abstractmethod
    def closed_form_pv(self, disc_curve):
        pass

    @abstractmethod
    def has_closed_form(self):
        return False
