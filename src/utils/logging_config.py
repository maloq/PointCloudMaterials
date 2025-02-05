import logging
from types import MethodType
from pytorch_lightning.utilities import rank_zero_info  # Make sure rank_zero_info is imported

def setup_logging(log_file='training.log'):
    """
    Configures the root logger to log messages to both stdout and a file.
    Clears existing handlers to avoid duplicate logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )

    logger = logging.getLogger()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    simple_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt="%H:%M:%S")
    file_handler.setFormatter(simple_formatter)
    logger.addHandler(file_handler)

    def custom_print(self, msg):
        rank_zero_info(msg)
    logger.print = MethodType(custom_print, logger)

    return logger