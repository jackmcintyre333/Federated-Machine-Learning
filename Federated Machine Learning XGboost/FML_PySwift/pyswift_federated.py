import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List
from sklearn.metrics import confusion_matrix

class FederatedClient:
    def __init__(self, client_id: int, data: pd.DataFrame, label_col: str):
        self.client_id = client_id
        self.data = data
        self.label_col = label_col
        self.model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

    def train_local_model(self):
        X = self.data.drop(columns=[self.label_col])
        y = self.data[self.label_col]
        self.model.fit(X, y)

    def get_model(self):
        return self.model

class PySwiftFederatedFramework:
    def __init__(self, df: pd.DataFrame, label_col: str, num_clients: int):
        self.label_col = label_col
        self.clients = self._split_into_clients(df, num_clients)

    def _split_into_clients(self, df: pd.DataFrame, num_clients: int) -> List[FederatedClient]:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        chunks = np.array_split(df_shuffled, num_clients)
        return [FederatedClient(i, chunk, self.label_col) for i, chunk in enumerate(chunks)]

    def run_federated_training(self):
        models = []
        for client in self.clients:
            print(f"Training on client {client.client_id}")
            client.train_local_model()
            models.append(client.get_model())

        # For MVP, just return all models
        return models

    # Add this new method to your class
    def evaluate_fedavg_model(self, test_data: pd.DataFrame):
        X_test = test_data.drop(columns=[self.label_col])
        y_test = test_data[self.label_col]

        # Collect all client predictions (probabilities)
        all_probs = []
        for client in self.clients:
            model = client.get_model()
            probas = model.predict_proba(X_test)
            all_probs.append(probas)

        # Average probabilities (FedAvg simulation)
        avg_probs = np.mean(np.array(all_probs), axis=0)
        avg_preds = np.argmax(avg_probs, axis=1)

        acc = accuracy_score(y_test, avg_preds)
        prec = precision_score(y_test, avg_preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, avg_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, avg_preds, average='weighted', zero_division=0)

        print(f"\n🔁 Federated Averaged Model Evaluation:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        cm = confusion_matrix(y_test, avg_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print("  Confusion Matrix (Binary):")
            print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        else:
            print("  Confusion Matrix (Multi-class):")
            print(cm)
