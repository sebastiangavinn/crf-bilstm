import logging
from pathlib import Path

def setup_logger(log_path: str):
    """
    Setup logger to write logs into a file.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
    )

    return logging.getLogger()
