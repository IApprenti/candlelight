"""Contains some helper functions"""

import os
import pprint
import time
import uuid
from datetime import datetime
from typing import Any
import random


prefix_template: str = "%Y-%m-%d_%H-%M-%S"


def append_to_file(message: str, file_path: str):
    """Append a message to a file"""
    with open(file_path, "a", encoding='UTF-8') as file:
        file.write(f"\n{message}")


def save_to_file(obj: Any, file_path: str):
    """Print the object to a file"""
    with open(file_path, "a", encoding='UTF-8') as file:
        pprint.pprint(obj, stream=file)


def current_timestamp() -> float:
    """Return the current timestamp"""
    return time.time()


def timestamp_to_prefix(timestamp: float) -> str:
    """Convert a timestamp to a prefix"""
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object.strftime(prefix_template)


def current_timestamp_as_prefix() -> str:
    """Return the current timestamp as a prefix"""
    timestamp_to_prefix(current_timestamp())


def now_to_prefix(now: datetime | None = None) -> str:
    """Return the current datetime as a prefix"""
    now_datetime: datetime = datetime.now() if now is None else now
    return now_datetime.strftime(prefix_template)


def create_uuid() -> str:
    """Create a unique identifier"""
    return str(uuid.uuid4())


def create_directory(directory: str) -> None:
    """Create the directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def random_bool() -> bool:
    """Randomly return True or False"""
    return bool(random.choice([True, False]))
