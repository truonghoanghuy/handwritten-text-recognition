import shutil


class ProgressBarPrinter:
    def __init__(self, num_iteration: int, progress_bar_length: int = 50):
        if progress_bar_length <= 0:
            raise ValueError('progress_bar_length must be positive')
        self.__count__ = 0
        self.__num_iter__ = num_iteration
        self.__progress_bar_length__ = progress_bar_length
        self.__pattern__ = '\r[{:-<{width}}] {}/{} ({}%)'
        self.__skip_count__ = 0
        self.__skip_pattern__ = ' - Skipped {}'

    def step(self, skip=False):
        if self.__count__ < self.__num_iter__:
            self.__count__ += 1
            if skip:
                self.__skip_count__ += 1
            percentage = int(self.__count__ / self.__num_iter__ * 100)
            passed_count = int(self.__count__ / self.__num_iter__ * self.__progress_bar_length__)
            output = self.__pattern__.format('=' * (passed_count - 1) + '>' * (passed_count > 0),
                                             self.__count__,
                                             self.__num_iter__,
                                             int(percentage),
                                             width=self.__progress_bar_length__)
            if self.__skip_count__ > 0:
                output += self.__skip_pattern__.format(self.__skip_count__)
            print('{: <{width}}'.format(output, width=shutil.get_terminal_size().columns), end='\r')
            if self.__count__ == self.__num_iter__:
                print(flush=True)

    def start(self):
        self.__count__ = -1
        self.step()
