import os
import joblib
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def create_ids_model(model_name):
    """
    Factory function returning the sklearn model based on model_name.
    """
    model_name = model_name.upper()
    if model_name == "SVM":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base_svm = LinearSVC(random_state=42, max_iter=2000)
        return CalibratedClassifierCV(base_svm, cv=3)
    elif model_name == "NB":
        return GaussianNB()
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    elif model_name == "LR":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "DT":
        return DecisionTreeClassifier(random_state=42)
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train_ids_model(model, X_train, y_train):
    """
    Train the model.
    """
    model.fit(X_train, y_train)
    return model

def predict_ids(model, X):
    """
    Returns binary predictions (0=normal, 1=malicious)
    """
    return model.predict(X)

def query_ids(model, X):
    """
    Same as predict but named to match paper terminology.
    Simulates black-box querying.
    """
    return predict_ids(model, X)

def evaluate_ids(model, X_test, y_test):
    """
    Returns dict with accuracy, precision, recall, f1
    """
    y_pred = predict_ids(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }

def save_ids_model(model, model_name, path):
    """
    Save the model using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_ids_model(model_name, path):
    """
    Load the model using joblib.
    """
    return joblib.load(path)
