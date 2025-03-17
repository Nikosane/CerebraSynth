import logging

def setup_logger(name="CerebraSynth", log_file="training.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()
