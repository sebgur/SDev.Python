

class MarketState:
    def __init__(self, disc_paths, event_paths, discount_curve):
        self.disc_paths = disc_paths
        self.event_paths = event_paths
        # self.terminal_spots = paths[:, -1, :]
        self.discount_curve = discount_curve
        self.n_paths = disc_paths.shape[0]
