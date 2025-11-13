import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Creating preprocessing object")

            numerical_features = [
                "Product_Price", "Order_Quantity", "Days_to_Return", "User_Age",
                "Discount_Applied", "Delivery_Days", "Past_Orders", "Past_Returns", "Past_Return_Rate"
            ]

            categorical_features = [
                "Product_Category", "Return_Reason", "User_Gender", "User_Location",
                "Payment_Method", "Shipping_Method", "Region"
            ]

            # Numeric pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessing_obj = self.get_data_transformation_object()

            target_column = "Return_Status"
            drop_columns = ["Order_ID", "Product_ID", "User_ID", "Order_Date", "Return_Date"]

            X_train = train_df.drop(columns=drop_columns + [target_column], errors='ignore')
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=drop_columns + [target_column], errors='ignore')
            y_test = test_df[target_column]

            X_train_scaled = preprocessing_obj.fit_transform(X_train)
            X_test_scaled = preprocessing_obj.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)
