import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a serialized object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
