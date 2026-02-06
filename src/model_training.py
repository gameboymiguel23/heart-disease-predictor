from dataclasses import dataclass
from typing import Dict, Any

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelTrainer:
    model_type: str = "logreg"  # "logreg" or "rf"
    random_state: int = 42

    def build_model(self) -> Pipeline:
        if self.model_type == "logreg":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000))
            ])

        if self.model_type == "rf":
            return Pipeline([
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    class_weight="balanced"
                ))
            ])

        raise ValueError("model_type must be 'logreg' or 'rf'")

    def train(self, X, y, test_size: float = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        model = self.build_model()
        model.fit(X_train, y_train)
        return model, X_test, y_test

    def evaluate(self, model: Pipeline, X_test, y_test) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        results: Dict[str, Any] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            results["roc_auc"] = roc_auc_score(y_test, y_proba)

        return results

    def save(self, model: Pipeline, filepath: str = "model.joblib") -> None:
        joblib.dump(model, filepath)

    def load(self, filepath: str = "model.joblib") -> Pipeline:
        return joblib.load(filepath)
