import os
import sys
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "New_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, preprocessor):
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }

            best_model = None
            best_name = None
            best_score = 0

            for name, model in models.items():
                logging.info(f"Training model: {name}")

                pipe = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(y_test, y_pred)

                logging.info(f"{name} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
                logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
                logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                if f1 > best_score:
                    best_model = pipe
                    best_name = name
                    best_score = f1

            # Save best model
            save_object(file_path=self.config.trained_model_file_path, obj=best_model)
            logging.info(f"Best model saved: {best_name}")

            print("\n🏆 Best Model:", best_name)
            print(f"🎯 F1 Score: {best_score:.4f}")
            print(f"📁 Model saved at: {self.config.trained_model_file_path}")

            return best_name, best_score

        except Exception as e:
            raise CustomException(e, sys)
