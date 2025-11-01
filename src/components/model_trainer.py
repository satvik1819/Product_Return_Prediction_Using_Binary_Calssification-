# src/components/model_trainer.py
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        """Train, tune and evaluate multiple models."""
        try:
            report = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                param_grid = params.get(model_name, {})
                if param_grid:
                    gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report[model_name] = acc
                trained_models[model_name] = model

                logging.info(f"{model_name} Accuracy: {acc:.4f}")
                logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            return report, trained_models

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_df, test_df):
        try:
            logging.info("Model training initiated")

            # ❌ Drop only columns not useful for training
            drop_cols = ['Order_ID', 'Product_ID', 'User_ID', 'Order_Date', 'Return_Date']
            train_df = train_df.drop(columns=drop_cols, errors='ignore')
            test_df = test_df.drop(columns=drop_cols, errors='ignore')

            # 🎯 Target column
            target_column = 'Return_Status'

            # ✅ Separate features (X) and target (y)
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # 🟩 Categorical and 🟦 numerical columns
            cat_cols = [
                'Product_Category', 'Return_Reason', 'User_Gender',
                'User_Location', 'Payment_Method', 'Shipping_Method', 'Region'
            ]
            num_cols = [
                'Product_Price', 'Order_Quantity', 'Days_to_Return',
                'User_Age', 'Discount_Applied', 'Delivery_Days',
                'Past_Orders', 'Past_Returns', 'Past_Return_Rate'
            ]

            # 🧠 Preprocessor
            preprocessor = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
                ('num', StandardScaler(), num_cols)
            ])

            # ⚙️ Model pipelines
            models = {
                "Logistic Regression": Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(max_iter=1000))
                ]),
                "Decision Tree": Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', DecisionTreeClassifier())
                ]),
                "Random Forest": Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier())
                ]),
                "Gradient Boosting": Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', GradientBoostingClassifier())
                ]),
                "SVM": Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', SVC())
                ])
            }

            # 🔍 Hyperparameter tuning
            params = {
                "Random Forest": {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 10, 20]
                },
                "Gradient Boosting": {
                    "classifier__n_estimators": [100, 150],
                    "classifier__learning_rate": [0.05, 0.1]
                }
            }

            # 🧪 Train & evaluate
            model_report, trained_models = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # 🏆 Pick best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_model_score:.4f}")

            # 💾 Save the trained pipeline
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            print("\n✅ Best Model:", best_model_name)
            print(f"🎯 Accuracy: {best_model_score:.4f}")
            y_pred = best_model.predict(X_test)
            print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
