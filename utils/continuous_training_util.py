import os


class ContinuousTrainingUtil:
    FILENAME = '.running'

    @staticmethod
    def start():
        open(ContinuousTrainingUtil.FILENAME, 'w+').close()

    @staticmethod
    def stop():
        try:
            os.remove(ContinuousTrainingUtil.FILENAME)
        except OSError:
            pass

    @staticmethod
    def is_running():
        return os.path.exists(ContinuousTrainingUtil.FILENAME)
