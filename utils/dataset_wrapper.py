from torch.utils.data import DataLoader


class DatasetWrapper:
    """
    Wrapper class to create view epoch for validation process.
    Validation is performed each ``view_epoch_size`` samples instead of at the end of an epoch.
    """

    def __init__(self, dataset: DataLoader, view_epoch_size: int):
        self.__dataset__ = dataset
        self.__view_epoch_size__ = view_epoch_size
        self.__current_index__ = 0
        self.__iter_dataset__ = iter(dataset)
        self.epoch = 0
        self.epoch_steps = len(dataset)
        self.batch_size = dataset.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.__current_index__ >= self.__view_epoch_size__:
            self.__current_index__ = 0
            raise StopIteration
        self.__current_index__ += 1
        try:
            return next(self.__iter_dataset__)
        except StopIteration:
            return self.__next_epoch__()

    def __next_epoch__(self):
        self.__iter_dataset__ = iter(self.__dataset__)
        self.epoch += 1
        return next(self.__iter_dataset__)

    def __len__(self):
        return self.__view_epoch_size__
