import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

class ESR1OnlyClassifier(BaseEstimator, ClassifierMixin):
    """Logistic regression classifier using ESR1 expression exclusively."""
    def __init__(self, esr1_idx=None, penalty="l2", C=1.0, random_state=42):
        self.esr1_idx = esr1_idx
        self.penalty = penalty
        self.C = C
        self.random_state = random_state
        self.model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.random_state
        )
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.esr1_idx is not None:
            X_feat = X[:, [self.esr1_idx]]
        else:
            X_feat = X
        self.model.fit(X_feat, y)
        return self

    def predict(self, X):
        X_feat = X[:, [self.esr1_idx]] if self.esr1_idx is not None else X
        return self.model.predict(X_feat)

    def predict_proba(self, X):
        X_feat = X[:, [self.esr1_idx]] if self.esr1_idx is not None else X
        return self.model.predict_proba(X_feat)


class SSMLPNet(nn.Module):
    """Compact two-hidden-layer neural network for ER status prediction."""
    def __init__(self, in_features, hidden1=256, dropout1=0.35, hidden2=64, dropout2=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SSMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for PyTorch SS-MLP architecture."""
    def __init__(
        self,
        in_features=None,
        hidden1=256,
        dropout1=0.35,
        hidden2=64,
        dropout2=0.25,
        learning_rate=0.0005,
        weight_decay=0.001,
        batch_size=32,
        max_epochs=100,
        patience=12,
        device=None,
        random_state=42
    ):
        self.in_features = in_features
        self.hidden1 = hidden1
        self.dropout1 = dropout1
        self.hidden2 = hidden2
        self.dropout2 = dropout2
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            try:
                # Test if GPU architecture is supported by PyTorch kernels (e.g. sm_120 Blackwell)
                test_tensor = torch.zeros(1, device="cuda") + 1.0
                return torch.device("cuda")
            except Exception:
                return torch.device("cpu")
        return torch.device("cpu")

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        
        in_dim = X.shape[1] if self.in_features is None else self.in_features
        self.model_ = SSMLPNet(
            in_features=in_dim,
            hidden1=self.hidden1,
            dropout1=self.dropout1,
            hidden2=self.hidden2,
            dropout2=self.dropout2
        )
        
        dev = self._get_device()
        self.model_.to(dev)

        # Compute positive class weight from training fold
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        train_dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        use_val = X_val is not None and y_val is not None
        if use_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
            y_val_t = torch.tensor(y_val, dtype=torch.float32, device=dev)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            for bx, by in train_loader:
                bx, by = bx.to(dev), by.to(dev)
                optimizer.zero_grad()
                logits = self.model_(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

            # Evaluation for early stopping
            if use_val:
                self.model_.eval()
                with torch.no_grad():
                    val_logits = self.model_(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
            self.model_.to(dev)
            
        return self

    def predict_proba(self, X):
        self.model_.eval()
        dev = self._get_device()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32, device=dev)
            logits = self.model_(x_tensor)
            probs_pos = torch.sigmoid(logits).cpu().numpy().reshape(-1, 1)
            probs_neg = 1.0 - probs_pos
        return np.hstack([probs_neg, probs_pos])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def get_model(name, esr1_idx=None, random_state=42):
    """Instantiate a baseline or proposed model with standardized hyperparameter configurations."""
    if name == "esr1_logistic":
        return ESR1OnlyClassifier(esr1_idx=esr1_idx, penalty="l2", C=1.0, random_state=random_state)
    elif name == "logistic_regression":
        return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=random_state)
    elif name == "elastic_net":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1, class_weight="balanced", max_iter=2000, random_state=random_state)
    elif name == "svm_linear":
        base_svm = SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=random_state)
        return CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
    elif name == "svm_rbf":
        base_svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=random_state)
        return CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    elif name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=random_state, eval_metric="logloss", n_jobs=-1)
    elif name == "ss_mlp":
        return SSMLPClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {name}")
