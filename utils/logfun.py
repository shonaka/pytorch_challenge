from logging import StreamHandler, INFO, DEBUG, Formatter, FileHandler, getLogger
from pathlib import Path

def set_logger(save_output, log_file_name):
    """
    Obtained and modified from Best practices to log from CS230 Deep Learning, Stanford.
    https://cs230-stanford.github.io/logging-hyperparams.html
    :param save_output: The directory of where you want to save the logs
    :param log_file_name: The name of the log file
    """
    logger = getLogger()
    logger.setLevel(INFO)

    if not logger.handlers:
        print("Calling the custom logger")
        # Define settings for logging
        log_format = Formatter(
            '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
        # for streaming, up to INFO level
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # for file, up to DEBUG level
        handler = FileHandler(save_output + '/' + log_file_name, 'a')
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)

    return logger


# Record time and print hours, mins, seconds
def timer(start, end):
    """
    Convert the measured time into hours, mins, seconds and print it out.
    :param start: start time (from import time, use start = time.time())
    :param end: end time (from import time, use end = time.time()
    :return: the time spent in hours, mins, seconds
    """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return int(hours), int(minutes), seconds
