import pandas as pd

import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import os
import json
from datetime import datetime
from database import get_db
import logging
logger = logging.getLogger('stride.federation')

BASE = r"C:\Users\ASUS\Documents\MBZUAI Project"

# =========================
# MODEL DEFINITION
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4,
                 num_layers=3, dim_feedforward=256,
                 dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerEncoder(
            dec_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model))
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, input_dim))

    def forward(self, x):
        x = self.pos_encoding(self.input_proj(x))
        return self.output_proj(
            self.decoder(self.encoder(x)))

# =========================
# ORGANIZATION CLASS
# Each org has its own
# local model + data
# =========================

class Organization:
    def __init__(self, org_id, name, model_path,
                 data_size, description):
        self.org_id = org_id
        self.name = name
        self.data_size = data_size
        self.description = description
        self.model = TransformerAutoencoder(
            input_dim=78)
        self.auc_before = None
        self.auc_after = None

        # Load model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(
                model_path, weights_only=False))
            self.model.eval()
            print(f"Org {name}: model loaded from "
                  f"{model_path}")
        else:
            print(f"Org {name}: model not found at "
                  f"{model_path}, using random weights")

    def get_weights(self):
        """Return model weights for federation"""
        return {k: v.clone()
                for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        """Update model with federated weights"""
        self.model.load_state_dict(weights)
        self.model.eval()

    def evaluate(self, X_test, y_test,seq_len=5, batch_size=64):
        """Evaluate model on test data.
        Uses flow-level scoring for better class balance.
        """
        from sklearn.metrics import roc_auc_score

        if len(X_test) == 0:
            return 0.5

        # Flow-level evaluation — score each flow directly
        # by treating each flow as a sequence of 1
        self.model.eval()
        all_errors = []

        # Process in small sequences
        n = len(X_test)
        for i in range(0, n - seq_len + 1, seq_len):
            batch = torch.FloatTensor(
                X_test[i:i+seq_len]).unsqueeze(0)
            with torch.no_grad():
                x_hat = self.model(batch)
                error = torch.mean(
                    (batch - x_hat)**2).item()
            all_errors.append(error)

        if len(all_errors) == 0:
            return 0.5

        # Get labels for each sequence
        seq_labels = []
        for i in range(0, n - seq_len + 1, seq_len):
            chunk = y_test[i:i+seq_len]
            seq_labels.append(1 if chunk.sum() > 0 else 0)

        errors = np.array(all_errors)
        y_seq = np.array(seq_labels)

        n_pos = y_seq.sum()
        n_neg = len(y_seq) - n_pos

        print(f"  {self.name}: {len(y_seq)} seqs, "
            f"{n_pos} attack, {n_neg} benign")

        if n_pos == 0 or n_neg == 0:
            # Fall back to flow-level scoring
            print(f"  Falling back to flow-level scoring...")
            flow_errors = []
            for i in range(0, n - seq_len + 1, 1):
                batch = torch.FloatTensor(
                    X_test[i:i+seq_len]).unsqueeze(0)
                with torch.no_grad():
                    x_hat = self.model(batch)
                    error = torch.mean(
                        (batch - x_hat)**2).item()
                flow_errors.append(error)

            flow_labels = []
            for i in range(0, n - seq_len + 1, 1):
                flow_labels.append(int(y_test[i]))

            errors = np.array(flow_errors)
            y_seq = np.array(flow_labels)
            n_pos = y_seq.sum()
            n_neg = len(y_seq) - n_pos
            print(f"  Flow-level: {len(y_seq)} flows, "
                f"{n_pos} attack, {n_neg} benign")

            if n_pos == 0 or n_neg == 0:
                return 0.5

        # Normalize
        if errors.max() > errors.min():
            errors_norm = (errors - errors.min()) / \
                        (errors.max() - errors.min())
        else:
            errors_norm = errors

        try:
            auc = float(roc_auc_score(y_seq, errors_norm))
            print(f"  AUC = {auc:.4f}")
            return auc
        except Exception as e:
            print(f"  AUC error: {e}")
            return 0.5

# =========================
# FEDERATED AGGREGATOR
# Implements FedAvg
# =========================

class FederatedAggregator:
    def __init__(self):
        self.round_history = []
        self.global_model = TransformerAutoencoder(
            input_dim=78)
        self.is_running = False
        print("Federated aggregator initialized.")

    def fed_avg(self, org_weights, data_sizes):
        """
        FedAvg Algorithm
        Weighted average of model weights
        Weight proportional to data size

        Global = Σ(n_i / N * w_i)
        where n_i = org data size
              N = total data size
              w_i = org model weights
        """
        total_data = sum(data_sizes)
        global_weights = {}

        # Get all parameter names from first org
        param_names = list(org_weights[0].keys())

        for param_name in param_names:
            # Weighted sum
            weighted_sum = None
            for i, (weights, size) in enumerate(
                    zip(org_weights, data_sizes)):
                weight_contribution = \
                    weights[param_name].float() * \
                    (size / total_data)
                if weighted_sum is None:
                    weighted_sum = weight_contribution
                else:
                    weighted_sum += weight_contribution

            global_weights[param_name] = weighted_sum

        return global_weights

    def run_federation_round(self, organizations,
                              X_test, y_test):
        """
        Run one round of federated learning:
        1. Collect weights from all orgs
        2. Apply FedAvg
        3. Distribute global weights back
        4. Evaluate improvement
        """
        if self.is_running:
            return None

        self.is_running = True
        print(f"\n{'='*50}")
        print(f"FEDERATED LEARNING ROUND")
        print(f"Organizations: {len(organizations)}")
        print(f"{'='*50}")

        try:
            # Step 1: Evaluate before federation
            print("\nEvaluating orgs BEFORE federation...")
            aucs_before = []
            for org in organizations:
                auc = org.evaluate(X_test, y_test)
                org.auc_before = auc
                aucs_before.append(auc)
                print(f"  {org.name}: AUC = {auc:.4f}")

            avg_before = np.mean(aucs_before)
            print(f"  Average AUC before: {avg_before:.4f}")

            # Step 2: Collect weights
            print("\nCollecting model weights...")
            org_weights = [org.get_weights()
                          for org in organizations]
            data_sizes = [org.data_size
                         for org in organizations]
            print(f"  Data sizes: {data_sizes}")
            print(f"  Total data: {sum(data_sizes)}")

            # Step 3: FedAvg
            print("\nRunning FedAvg...")
            global_weights = self.fed_avg(
                org_weights, data_sizes)
            print("  FedAvg complete.")

            # Step 4: Update global model
            self.global_model.load_state_dict(
                global_weights)
            self.global_model.eval()

            # Step 5: Distribute to all orgs
            print("\nDistributing global model...")
            for org in organizations:
                org.set_weights(global_weights)

            # Step 6: Evaluate after federation
            print("\nEvaluating orgs AFTER federation...")
            aucs_after = []
            for org in organizations:
                auc = org.evaluate(X_test, y_test)
                org.auc_after = auc
                aucs_after.append(auc)
                print(f"  {org.name}: AUC = {auc:.4f} "
                      f"(was {org.auc_before:.4f}, "
                      f"Δ{auc-org.auc_before:+.4f})")

            avg_after = np.mean(aucs_after)
            improvement = avg_after - avg_before
            print(f"\n  Average AUC after:  {avg_after:.4f}")
            print(f"  Improvement: {improvement:+.4f}")

            # Step 7: Save global model
            global_path = f"{BASE}/transformer_federated.pth"
            torch.save(self.global_model.state_dict(),
                      global_path)
            print(f"\nGlobal model saved to {global_path}")

            # Step 8: Save round results
            round_result = {
                "timestamp": datetime.now().isoformat(),
                "round": len(self.round_history) + 1,
                "n_organizations": len(organizations),
                "orgs": [
                    {
                        "name": org.name,
                        "data_size": org.data_size,
                        "auc_before": round(
                            org.auc_before, 4),
                        "auc_after": round(
                            org.auc_after, 4),
                        "improvement": round(
                            org.auc_after -
                            org.auc_before, 4)
                    }
                    for org in organizations
                ],
                "avg_auc_before": round(float(avg_before), 4),
                "avg_auc_after": round(float(avg_after), 4),
                "improvement": round(float(improvement), 4),
                "fed_avg_weights": len(global_weights),
                "total_data": sum(data_sizes)
            }

            self.round_history.append(round_result)

            # Save to database
            self._save_to_db(round_result)

            self.is_running = False
            return round_result

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.is_running = False
            return {"error": str(e)}

    def _save_to_db(self, round_result):
        """Save federation round to database"""
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS federation_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                round_number INTEGER,
                n_organizations INTEGER,
                avg_auc_before REAL,
                avg_auc_after REAL,
                improvement REAL,
                details TEXT
            )
        ''')
        c.execute('''
            INSERT INTO federation_rounds
            (timestamp, round_number, n_organizations,
             avg_auc_before, avg_auc_after,
             improvement, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            round_result['timestamp'],
            round_result['round'],
            round_result['n_organizations'],
            round_result['avg_auc_before'],
            round_result['avg_auc_after'],
            round_result['improvement'],
            json.dumps(round_result)
        ))
        conn.commit()
        conn.close()
        print("Federation round saved to database.")

    def get_history(self):
        """Get federation round history"""
        conn = get_db()
        c = conn.cursor()
        try:
            c.execute('''
                SELECT * FROM federation_rounds
                ORDER BY timestamp DESC
                LIMIT 20
            ''')
            rows = c.fetchall()
            conn.close()
            results = []
            for r in rows:
                d = dict(r)
                if 'details' in d:
                    d['details'] = json.loads(
                        d['details'])
                results.append(d)
            return results
        except:
            conn.close()
            return []

# =========================
# FEDERATION MANAGER
# Sets up the 3 orgs and
# runs federation
# =========================

class FederationManager:
    def __init__(self):
        self.aggregator = FederatedAggregator()
        self.organizations = self._setup_orgs()
        self.scaler = joblib.load(
            f"{BASE}/scaler_transformer_v3.pkl")
        print(f"\nFederation Manager ready with "
              f"{len(self.organizations)} organizations.")

    def _setup_orgs(self):
        """
        Set up 3 simulated organizations:
        - Org A: UAE Bank (CICIDS V3 model)
        - Org B: Government (Robust model)
        - Org C: Telecom (Evolved model or V3)
        """
        # Org C uses evolved model if exists, else V3
        org_c_path = f"{BASE}/transformer_evolved.pth"
        if not os.path.exists(org_c_path):
            org_c_path = f"{BASE}/transformer_v3_model.pth"

        orgs = [
            Organization(
                org_id="org_a",
                name="UAE Bank (Org A)",
                model_path=f"{BASE}/transformer_v3_model.pth",
                data_size=2095057,
                description="Primary financial institution — "
                        "trained on CICIDS-2017"
            ),
            Organization(
                org_id="org_b",
                name="Government (Org B)",
                model_path=f"{BASE}/transformer_robust_model.pth",
                data_size=93000,
                description="Government network — "
                        "adversarially robust model"
            ),
            Organization(
                org_id="org_c",
                name="Telecom (Org C)",
                model_path=org_c_path,
                data_size=500000,
                description="Telecommunications provider — "
                        "self-evolved model"
            )
        ]
        return orgs

    def prepare_test_data(self, n_samples=5000):
        """
        Prepare balanced test data for evaluation.
        Ensures both benign and attack samples present.
        """
        import glob
        import pandas as pd

        print("Preparing balanced test data...")
        FOLDER = r"C:\Users\ASUS\Documents\MBZUAI Project\archive"

        files = glob.glob(os.path.join(FOLDER, "*.csv"))
        if not files:
            return None, None

            # Prioritize files with attacks
        ddos_files = [f for f in files
                    if 'DDos' in f or 'DDoS' in f]
        other_files = [f for f in files
                    if 'DDos' not in f and 'DDoS' not in f]
        ordered_files = ddos_files + other_files

        df_list = []
        for f in ordered_files[:5]:
            try:
                temp = pd.read_csv(
                    f, low_memory=False, nrows=20000)
                temp.columns = temp.columns.str.strip()
                df_list.append(temp)
                print(f"  Loaded {os.path.basename(f)}")
            except:
                continue

        if not df_list:
            return None, None

        df = pd.concat(df_list, ignore_index=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        # Separate benign and attacks
        benign = df[df['Label'] == 'BENIGN']
        attacks = df[df['Label'] != 'BENIGN']

        print(f"Available — Benign: {len(benign)}, "
            f"Attack: {len(attacks)}")

        if len(attacks) == 0:
            print("No attacks found!")
            return None, None

        if len(benign) == 0:
            print("No benign found!")
            return None, None

        # Strictly balanced — equal benign and attacks
        n_each = min(
            n_samples // 2,
            len(benign),
            len(attacks))
        
        n_each = max(n_each, min(50, len(attacks)))

        benign_sample = benign.sample(
            n=n_each, random_state=42)
        # Use all attacks if fewer than n_each
        if len(attacks) <= n_each:
            attack_sample = attacks
        else:
            attack_sample = attacks.sample(
                n=n_each, random_state=42)

        # Adjust benign to match
        n_benign = min(len(benign), len(attack_sample) * 10)
        benign_sample = benign.sample(
            n=n_benign, random_state=42)

        df_balanced = pd.concat(
            [benign_sample, attack_sample]).sample(
            frac=1, random_state=42).reset_index(drop=True)

        print(f"Balanced: {len(df_balanced)} rows "
            f"({n_each} benign + {n_each} attack)")

        y = (df_balanced['Label'] != 'BENIGN').astype(
            int).values
        X = df_balanced.drop(
            'Label', axis=1).select_dtypes(
            include=[np.number]).values

        # Align to 78 features
        if X.shape[1] < 78:
            padding = np.zeros(
                (X.shape[0], 78 - X.shape[1]))
            X = np.hstack([X, padding])
        else:
            X = X[:, :78]

        # Handle any remaining inf/nan
        X = np.nan_to_num(X, nan=0.0,
                        posinf=0.0, neginf=0.0)

        X = self.scaler.transform(X)

        # Final check
        unique_labels = np.unique(y)
        print(f"Labels in test set: {unique_labels}")
        print(f"Attack ratio: {y.mean()*100:.1f}%")

        if len(unique_labels) < 2:
            print("ERROR: Still only one class!")
            return None, None

        return X, y

    def run_federation(self):
        """Run a complete federation round"""
        if self.aggregator.is_running:
            return {"error": "Federation already running"}

        X_test, y_test = self.prepare_test_data()
        if X_test is None:
            return {"error": "Could not load test data"}

        result = self.aggregator.run_federation_round(
            self.organizations, X_test, y_test)
        return result

    def get_status(self):
        """Get current federation status"""
        history = self.aggregator.get_history()
        return {
            "n_organizations": len(self.organizations),
            "organizations": [
                {
                    "id": org.org_id,
                    "name": org.name,
                    "data_size": org.data_size,
                    "description": org.description,
                    "auc_before": org.auc_before,
                    "auc_after": org.auc_after
                }
                for org in self.organizations
            ],
            "total_rounds": len(history),
            "history": history,
            "is_running": self.aggregator.is_running,
            "global_model_exists": os.path.exists(
                f"{BASE}/transformer_federated.pth")
        }