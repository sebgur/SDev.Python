from abc import ABC, abstractmethod


class Simulator(ABC):
    @abstractmethod
    def build_paths(self, init_spot, init_vol, rng):
        pass
