import time


class Stopwatch:
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        # self.end = time.time()
        self.total_elapsed = 0.0

    def trigger(self):
        self.start = time.time()

    def stop(self):
        # self.end = time.time()
        current_elapsed = time.time() - self.start
        self.total_elapsed += current_elapsed

    def elapsed(self):
        return self.total_elapsed
        # return self.end - self.start

    def print(self):
        elapsed = self.elapsed()
        print('Runtime(' + self.name + '): %.1f' % elapsed + 's')

    def clear(self):
        self.total_elapsed = 0.0
