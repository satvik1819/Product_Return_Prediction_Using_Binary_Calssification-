# src/exception.py
import sys
import traceback
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    func_name = exc_tb.tb_frame.f_code.co_name
    line_no = exc_tb.tb_lineno

    # Full traceback text
    tb = ''.join(traceback.format_exception(type(error), error, exc_tb))

    error_message = (
        f"\n----- CUSTOM ERROR TRACE -----\n"
        f"Error Type      : {type(error).__name__}\n"
        f"File            : {file_name}\n"
        f"Function        : {func_name}\n"
        f"Line Number     : {line_no}\n"
        f"Message         : {str(error)}\n"
        f"Full Traceback  :\n{tb}\n"
        f"--------------------------------\n"
    )

    logging.error(error_message)
    return error_message


class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message


# ---------------------------------------------------------------
# 📌 SPECIFIC ERROR CLASSES FOR YOUR WEB APPLICATION + ML MODELS
# ---------------------------------------------------------------

class DataLoadingException(CustomException):
    """Raised when CSV / Excel / DB connection fails."""
    pass


class DataValidationException(CustomException):
    """Raised when input data contains NaN, wrong types, or missing columns."""
    pass


class ModelNotFoundException(CustomException):
    """Raised when .pkl model file is not found or corrupted."""
    pass


class ModelPredictionException(CustomException):
    """Raised when model.predict() fails."""
    pass


class APICallException(CustomException):
    """Raised when any API call fails."""
    pass


class ConfigException(CustomException):
    """Raised when YAML/JSON configuration load fails."""
    pass


class WebpageRenderException(CustomException):
    """Raised when Gradio / Flask / FastAPI fails to render."""
    pass


class EnvironmentException(CustomException):
    """Errors related to system environment, env variables, OS, etc."""
    pass


class FileHandlingException(CustomException):
    """Raised when reading/writing a file fails."""
    pass


class DatabaseException(CustomException):
    """Raised when DB connection / SQL query fails."""
    pass


class UserInputException(CustomException):
    """Raised when user enters invalid or unexpected values."""
    pass


class FeatureEngineeringException(CustomException):
    """Raised during pipeline transformation issues."""
    pass
