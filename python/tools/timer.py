""" Timer tools including Stopwatch class"""
import time


class Stopwatch:
    """ Define a timer with an ID and output elapsed time """
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        self.total_elapsed = 0.0

    def trigger(self):
        """ Start the timer """
        self.start = time.time()

    def stop(self):
        """ Stop the timer """
        current_elapsed = time.time() - self.start
        self.total_elapsed += current_elapsed

    def elapsed(self):
        """ Calculate the elapsed time """
        return self.total_elapsed

    def print(self):
        """ Print the elapsted time """
        elapsed = self.elapsed()
        print(f'Runtime({self.name}): {elapsed:.1f}s')

    def clear(self):
        """ Clear the timer """
        self.total_elapsed = 0.0
