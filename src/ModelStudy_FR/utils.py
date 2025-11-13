"""_summary_
This is a part of standard cookie-cutter tempalte.
"""

import logging

class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.setLevel(logging.DEBUG)
        self.propagate = False
        # Adding a console handler
        console_handler = ConsoleHandler()
        self.addHandler(console_handler)

        # Adding a file handler
        file_handler = CustomFileHandler()
        self.addHandler(file_handler)

class ConsoleHandler(logging.StreamHandler):
    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(level)

class CustomFileHandler(logging.FileHandler):
    def __init__(self):
        log_file = "logfile.log"
        super().__init__(log_file, encoding="UTF-8")
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(logging.INFO)
