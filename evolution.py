import torch
import torch.nn as nn
import numpy as np
import joblib
import math
import os
from datetime import datetime
from database import get_db, save_scan

BASE = r"C:\Users\ASUS\Documents\MBZUAI Project"

# =========================
# MODEL DEFINITIONS
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
                 num_layers=3, dim_feedforward=256, dropout=0.1):
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
        return self.output_proj(self.decoder(self.encoder(x)))

# =========================
# EVOLUTION ENGINE
# =========================

class ModelEvolutionEngine:
    def __init__(self, input_dim=78, seq_len=20):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.scaler = joblib.load(
            f"{BASE}/scaler_transformer_v3.pkl")
        self.evolution_count = 0
        self.feedback_threshold = 10
        self.is_evolving = False

        # Load base model
        self.model = TransformerAutoencoder(
            input_dim=input_dim)
        self.model.load_state_dict(torch.load(
            f"{BASE}/transformer_v3_model.pth",
            weights_only=False))
        self.model.eval()
        print("Evolution engine initialized.")

    def get_pending_feedback(self):
        """Get all unprocessed feedback from database"""
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            SELECT d.*, s.filename
            FROM detections d
            JOIN scans s ON d.scan_id = s.id
            WHERE d.feedback IS NOT NULL
            ORDER BY d.feedback_timestamp ASC
        ''')
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_evolution_stats(self):
        """Get current evolution statistics"""
        conn = get_db()
        c = conn.cursor()

        # Total feedback
        c.execute('''
            SELECT feedback, COUNT(*) as count
            FROM detections
            WHERE feedback IS NOT NULL
            GROUP BY feedback
        ''')
        feedback_rows = c.fetchall()

        # Evolution history
        c.execute('''
            SELECT * FROM model_evolution
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        evolution_rows = c.fetchall()

        conn.close()

        feedback_stats = {r['feedback']: r['count']
                         for r in feedback_rows}
        evolution_history = [dict(r) for r in evolution_rows]

        total_feedback = sum(feedback_stats.values())
        confirmed_attacks = feedback_stats.get(
            'confirmed_attack', 0)
        false_positives = feedback_stats.get(
            'false_positive', 0)

        # Calculate precision from feedback
        precision = round(
            confirmed_attacks /
            (confirmed_attacks + false_positives + 1e-8),
            4) if (confirmed_attacks + false_positives) > 0 \
            else None

        return {
            'total_feedback': total_feedback,
            'confirmed_attacks': confirmed_attacks,
            'false_positives': false_positives,
            'confirmed_benign': feedback_stats.get(
                'confirmed_benign', 0),
            'false_negatives': feedback_stats.get(
                'false_negative', 0),
            'precision_from_feedback': precision,
            'evolution_count': self.evolution_count,
            'is_evolving': self.is_evolving,
            'ready_to_evolve': total_feedback >=
                               self.feedback_threshold,
            'feedback_until_evolution':
                max(0, self.feedback_threshold - total_feedback),
            'evolution_history': evolution_history
        }

    def evolve(self):
        """
        Core evolution function.
        Uses feedback to fine-tune the model.

        Strategy:
        - Confirmed attacks: increase sensitivity
          (lower reconstruction threshold)
        - False positives: decrease sensitivity
          (raise reconstruction threshold)
        - Use gradient updates on feedback examples
        """
        if self.is_evolving:
            print("Evolution already in progress.")
            return False

        feedback = self.get_pending_feedback()
        if len(feedback) < self.feedback_threshold:
            print(f"Not enough feedback yet. "
                  f"Need {self.feedback_threshold}, "
                  f"have {len(feedback)}")
            return False

        print(f"\n{'='*50}")
        print(f"STARTING MODEL EVOLUTION")
        print(f"Feedback items: {len(feedback)}")
        print(f"{'='*50}")

        self.is_evolving = True

        try:
            # Separate feedback by type
            confirmed_attacks = [f for f in feedback
                                 if f['feedback'] ==
                                 'confirmed_attack']
            false_positives = [f for f in feedback
                               if f['feedback'] ==
                               'false_positive']
            confirmed_benign = [f for f in feedback
                                if f['feedback'] ==
                                'confirmed_benign']

            print(f"Confirmed attacks: {len(confirmed_attacks)}")
            print(f"False positives: {len(false_positives)}")
            print(f"Confirmed benign: {len(confirmed_benign)}")

            # Fine-tune model
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-5,  # Very small LR for fine-tuning
                weight_decay=1e-4)

            self.model.train()
            total_loss = 0
            update_count = 0

            # For confirmed attacks: train model to give
            # HIGH reconstruction error (maximize loss)
            # We do this by using gradient ASCENT
            for item in confirmed_attacks:
                try:
                    # Create synthetic sequence from scores
                    scores = torch.FloatTensor([[
                        item['transformer_score'],
                        item['mlp_score'],
                        item['robust_score'],
                        item['ensemble_score']
                    ]])

                    # Pad to input dim and create sequence
                    padded = torch.zeros(1, self.seq_len,
                                        self.input_dim)
                    padded[0, :, :4] = scores.repeat(
                        self.seq_len, 1)

                    optimizer.zero_grad()
                    x_hat = self.model(padded)

                    # MAXIMIZE reconstruction error
                    # for confirmed attacks
                    loss = -torch.mean((padded - x_hat)**2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 0.5)
                    optimizer.step()

                    total_loss += abs(loss.item())
                    update_count += 1

                except Exception as e:
                    print(f"Skip attack item: {e}")
                    continue

            # For false positives: train model to give
            # LOW reconstruction error (minimize loss)
            for item in false_positives:
                try:
                    scores = torch.FloatTensor([[
                        item['transformer_score'],
                        item['mlp_score'],
                        item['robust_score'],
                        item['ensemble_score']
                    ]])

                    padded = torch.zeros(1, self.seq_len,
                                        self.input_dim)
                    padded[0, :, :4] = scores.repeat(
                        self.seq_len, 1)

                    optimizer.zero_grad()
                    x_hat = self.model(padded)

                    # MINIMIZE reconstruction error
                    # for false positives
                    loss = torch.mean((padded - x_hat)**2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 0.5)
                    optimizer.step()

                    total_loss += loss.item()
                    update_count += 1

                except Exception as e:
                    print(f"Skip fp item: {e}")
                    continue

            self.model.eval()

            # Save evolved model
            evolved_path = f"{BASE}/transformer_evolved.pth"
            torch.save(self.model.state_dict(), evolved_path)

            self.evolution_count += 1

            # Calculate metrics
            avg_loss = total_loss / (update_count + 1e-8)
            precision = len(confirmed_attacks) / (
                len(confirmed_attacks) +
                len(false_positives) + 1e-8)

            # Save evolution record to database
            conn = get_db()
            c = conn.cursor()
            c.execute('''
                INSERT INTO model_evolution
                (timestamp, feedback_count,
                 true_positives, false_positives,
                 true_negatives, false_negatives,
                 precision, recall, f1, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                len(feedback),
                len(confirmed_attacks),
                len(false_positives),
                len(confirmed_benign),
                0,
                round(float(precision), 4),
                0.0,
                0.0,
                f"Evolution #{self.evolution_count} — "
                f"{update_count} gradient updates — "
                f"avg loss: {avg_loss:.4f}"
            ))
            conn.commit()
            conn.close()

            print(f"\nEvolution #{self.evolution_count} complete!")
            print(f"Updates: {update_count}")
            print(f"Avg loss: {avg_loss:.4f}")
            print(f"Model saved to: {evolved_path}")

            self.is_evolving = False
            return True

        except Exception as e:
            print(f"Evolution failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_evolving = False
            return False

    def load_evolved_model(self):
        """Load the evolved model if it exists"""
        evolved_path = f"{BASE}/transformer_evolved.pth"
        if os.path.exists(evolved_path):
            self.model.load_state_dict(torch.load(
                evolved_path, weights_only=False))
            self.model.eval()
            print("Loaded evolved model.")
            return True
        return False

    def get_model(self):
        return self.model