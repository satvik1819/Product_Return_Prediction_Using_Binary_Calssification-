# src/components/data_transformation.py
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import TransformerMixin, BaseEstimator

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_obj_file_path = os.path.join("artifacts", "label_encoder.pkl")


class DateFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    Custom transformer to extract features from Order_Date and Return_Date columns.
    Expects DataFrame input and returns DataFrame with new features and dropped raw date cols.
    """
    def __init__(self, order_date_col="Order_Date", return_date_col="Return_Date"):
        self.order_date_col = order_date_col
        self.return_date_col = return_date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Parse Order_Date
        if self.order_date_col in X.columns:
            X[self.order_date_col] = pd.to_datetime(X[self.order_date_col], errors="coerce")
            X["order_month"] = X[self.order_date_col].dt.month.fillna(-1).astype(int)
            X["order_dayofweek"] = X[self.order_date_col].dt.dayofweek.fillna(-1).astype(int)
        else:
            X["order_month"] = -1
            X["order_dayofweek"] = -1

        # Return_Date: create boolean feature indicating presence and drop the raw column
        if self.return_date_col in X.columns:
            # Some Return_Date values are numeric/null; try parse where possible
            X[self.return_date_col] = pd.to_datetime(X[self.return_date_col], errors="coerce")
            X["has_return_date"] = (~X[self.return_date_col].isna()).astype(int)
            # Optionally, days between Order_Date and Return_Date (if both present)
            try:
                X["days_between_order_and_return"] = (
                    (X[self.return_date_col] - X[self.order_date_col]).dt.days
                ).fillna(-1).astype(float)
            except Exception:
                X["days_between_order_and_return"] = -1.0
        else:
            X["has_return_date"] = 0
            X["days_between_order_and_return"] = -1.0

        # Drop raw date columns - we've encoded them
        drop_cols = [self.order_date_col, self.return_date_col]
        for c in drop_cols:
            if c in X.columns:
                X.drop(columns=[c], inplace=True)

        return X


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the ColumnTransformer preprocessor that handles:
         - numerical imputation + scaling
         - categorical imputation + one-hot encoding + optional scaling
         - date feature extraction (custom transformer)
        """
        try:
            logging.info("Creating data transformer object")

            # Numerical columns (use all numeric-like features present in your schema)
            numerical_columns = [
                "Product_Price", "Order_Quantity", "Days_to_Return",
                "User_Age", "Discount_Applied", "Delivery_Days",
                "Past_Orders", "Past_Returns", "Past_Return_Rate",
                "days_between_order_and_return"  # created by DateFeatureExtractor
            ]

            # Categorical columns
            categorical_columns = [
                "Product_Category", "Return_Reason", "User_Gender",
                "User_Location", "Payment_Method", "Shipping_Method", "Region",
                "order_month", "order_dayofweek", "has_return_date"  # created by DateFeatureExtractor
            ]

            # NUMERIC pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # CATEGORICAL pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),

                    # scaling after one-hot is optional; with sparse=False safe to scale
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combine everything. Note: DateFeatureExtractor is applied separately before ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="drop"  # drop any other columns (IDs etc.)
            )

            logging.info("Preprocessor (ColumnTransformer) created.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads train/test CSV paths, applies transformations, encodes target,
        returns numpy arrays (X+y combined) and paths to saved objects.
        """
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test files read into DataFrames")

            # Keep a copy for debugging if needed
            train_df_original = train_df.copy()
            test_df_original = test_df.copy()

            # Drop identifier columns (they are not useful as features)
            id_cols = ["Order_ID", "Product_ID", "User_ID"]
            train_df.drop(columns=[c for c in id_cols if c in train_df.columns], inplace=True, errors="ignore")
            test_df.drop(columns=[c for c in id_cols if c in test_df.columns], inplace=True, errors="ignore")
            logging.info(f"Dropped identifier columns: {id_cols}")

            # Extract date-based features using custom transformer (applies to DataFrame)
            date_extractor = DateFeatureExtractor(order_date_col="Order_Date", return_date_col="Return_Date")
            train_df = date_extractor.transform(train_df)
            test_df = date_extractor.transform(test_df)
            logging.info("Date features extracted")

            # Target encoding: fit label encoder on training target and transform both
            target_col = "Return_Status"
            if target_col not in train_df.columns:
                raise CustomException(f"Target column '{target_col}' not found in training data", sys)

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_df[target_col].astype(str))
            y_test = label_encoder.transform(test_df[target_col].astype(str))

            # Drop target column from features
            X_train = train_df.drop(columns=[target_col], axis=1, errors="ignore")
            X_test = test_df.drop(columns=[target_col], axis=1, errors="ignore")
            logging.info("Separated input features and target")

            # Build preprocessor and fit/transform
            preprocessor = self.get_data_transformer_object()

            # Ensure all expected columns exist in X_train/X_test by adding missing ones with NaNs
            # We will rely on ColumnTransformer to select columns present; missing columns in the
            # transformer lists will cause errors — so make sure columns exist.
            # Add missing numerical and categorical columns as NaN columns so transformers can find them.
            # (This is defensive — modify lists in get_data_transformer_object if your schema changes)
            all_expected_cols = (
                ["Product_Price", "Order_Quantity", "Days_to_Return",
                 "User_Age", "Discount_Applied", "Delivery_Days",
                 "Past_Orders", "Past_Returns", "Past_Return_Rate",
                 "days_between_order_and_return"] +
                ["Product_Category", "Return_Reason", "User_Gender",
                 "User_Location", "Payment_Method", "Shipping_Method", "Region",
                 "order_month", "order_dayofweek", "has_return_date"]
            )
            for col in all_expected_cols:
                if col not in X_train.columns:
                    X_train[col] = np.nan
                if col not in X_test.columns:
                    X_test[col] = np.nan

            # Reorder columns so the transformers see columns in same order (not strictly necessary)
            X_train = X_train[sorted(X_train.columns)]
            X_test = X_test[sorted(X_test.columns)]

            # Fit-transform on training set, transform test set
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            logging.info("Applied preprocessing to train and test sets")

            # Combine features and labels
            train_arr = np.c_[X_train_arr, y_train]
            test_arr = np.c_[X_test_arr, y_test]

            # Save preprocessor and label encoder
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            save_object(file_path=self.config.label_encoder_obj_file_path, obj=label_encoder)
            logging.info(f"Saved preprocessor to {self.config.preprocessor_obj_file_path} "
                         f"and label encoder to {self.config.label_encoder_obj_file_path}")

            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
