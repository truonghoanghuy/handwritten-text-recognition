import numpy as np


class DotProgressPrinter(object):
    def __init__(self, num_iter, dot_length=80):
        if dot_length <= 0:
            raise ValueError('dot_length must be positive')
        progress_list = np.linspace(dot_length, 0, num_iter, endpoint=False)
        progress_list = np.flip(progress_list)
        self.progress = iter(progress_list)
        self.num_iter = num_iter
        self.count = 0

    def step(self):
        try:
            pointer = next(self.progress)
            while self.count < pointer:
                self.count += 1
                print('.', end='', flush=True)
                if self.count >= self.num_iter:
                    print()
        except StopIteration:
            pass
