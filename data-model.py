import argparse
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Train and log a water potability model.")
    parser.add_argument(
        "--data-path",
        default="data/water_potability.csv",
        help="Path to the water potability CSV dataset.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name to display for this MLflow run.",
    )
    return parser.parse_args()


args = parse_args()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("water-potability")

# Read dataset
data = pd.read_csv(args.data_path)


# Split the data into training and testing sets
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data.iloc[:, -1],
)


# Function to fill missing values with median
def fill_missing_with_median(df):

    df = df.copy()

    for column in df.columns:

        if df[column].isnull().any():

            median_value = df[column].median()

            df[column] = df[column].fillna(median_value)

    return df


# Fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


# Separate features and target
X_train = train_processed_data.iloc[:, :-1].values
y_train = train_processed_data.iloc[:, -1].values

X_test = test_processed_data.iloc[:, :-1].values
y_test = test_processed_data.iloc[:, -1].values


feature_names = train_processed_data.columns[:-1].tolist()


def log_metrics(prefix, y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    mlflow.log_metrics({f"{prefix}_{name}": value for name, value in metrics.items()})
    return metrics


# Start MLflow run
with mlflow.start_run(run_name=args.run_name):
    baseline_model = RandomForestClassifier(n_estimators=200, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_metrics = log_metrics("baseline", y_test, baseline_predictions)

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", 0.7],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    clf = search.best_estimator_
    y_pred = clf.predict(X_test)
    tuned_metrics = log_metrics("tuned", y_test, y_pred)
    mlflow.log_metric("cv_best_f1", search.best_score_)
    mlflow.log_params({f"best_{key}": value for key, value in search.best_params_.items()})
    mlflow.log_param("data_path", args.data_path)
    mlflow.set_tag("optimization_metric", "f1")

    balanced_model = BalancedRandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    balanced_model.fit(X_train, y_train)
    balanced_predictions = balanced_model.predict(X_test)
    balanced_metrics = log_metrics("balanced", y_test, balanced_predictions)
    mlflow.log_param("balanced_n_estimators", 300)
    mlflow.sklearn.log_model(balanced_model, "balanced-random-forest-model")

    balanced_confusion_matrix = confusion_matrix(y_test, balanced_predictions)
    ConfusionMatrixDisplay(balanced_confusion_matrix).plot(cmap="Greens")
    plt.title("Balanced Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("balanced_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("balanced_confusion_matrix.png")

    pickle.dump(clf, open("model.pkl", "wb"))
    mlflow.log_artifact("model.pkl")
    mlflow.sklearn.log_model(clf, "random-forest-model")

    confusion_matrix_values = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix_values).plot(cmap="Blues")
    plt.title("Tuned Water Potability Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    importance_order = np.argsort(clf.feature_importances_)
    plt.figure(figsize=(8, 5))
    plt.barh(
        np.array(feature_names)[importance_order],
        clf.feature_importances_[importance_order],
    )
    plt.title("Tuned Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()
    mlflow.log_artifact("feature_importance.png")

    print("baseline", baseline_metrics)
    print("tuned", tuned_metrics)
    print("balanced", balanced_metrics)
    print("best_params", search.best_params_)