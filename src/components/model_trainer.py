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
        """
        Train, tune and evaluate multiple models.
        Returns a tuple: (performance report, trained models dict)
        """
        try:
            report = {}
            trained_models = {}

            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                # Hyperparameter tuning
                param_grid = params.get(model_name, {})
                if param_grid:
                    logging.info(f"Performing GridSearchCV for {model_name}")
                    gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    logging.info(f"Best params for {model_name}: {gs.best_params_}")
                else:
                    model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report[model_name] = acc
                trained_models[model_name] = model  # ✅ store fitted version

                logging.info(f"{model_name} Accuracy: {acc:.4f}")
                logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            return report, trained_models

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training initiated")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC()
            }

            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 150],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5]
                },
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                }
            }

            # ✅ Evaluate and get both report + trained models
            model_report, trained_models = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # ✅ Pick the trained best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_model_score:.4f}")

            # Save the trained best model
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best model saved at {self.config.trained_model_file_path}")

            # Final evaluation
            y_pred = best_model.predict(X_test)
            print("\n-------------------------")
            print(f"✅ Best Model: {best_model_name}")
            print(f"🎯 Accuracy: {best_model_score:.4f}")
            print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)
