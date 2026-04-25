import sqlite3
import json
from datetime import datetime
import os

DB_PATH = r"C:\Users\ASUS\Documents\MBZUAI Project\stride\stride.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    # Scans table — stores every file analysis
    c.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            timestamp TEXT,
            total_sequences INTEGER,
            attack_sequences INTEGER,
            benign_sequences INTEGER,
            attack_rate REAL,
            avg_score REAL,
            max_score REAL,
            model_scores TEXT,
            status TEXT DEFAULT 'completed'
        )
    ''')

    # Detections table — stores individual sequence detections
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER,
            sequence_num INTEGER,
            transformer_score REAL,
            mlp_score REAL,
            robust_score REAL,
            ensemble_score REAL,
            prediction TEXT,
            risk_level TEXT,
            attack_type TEXT,
            feedback TEXT DEFAULT NULL,
            feedback_timestamp TEXT DEFAULT NULL,
            FOREIGN KEY (scan_id) REFERENCES scans(id)
        )
    ''')

    # Threat DNA table — stores attack fingerprints
    c.execute('''
        CREATE TABLE IF NOT EXISTS threat_dna (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dna_hash TEXT UNIQUE,
            attack_type TEXT,
            avg_transformer_score REAL,
            avg_mlp_score REAL,
            avg_ensemble_score REAL,
            occurrence_count INTEGER DEFAULT 1,
            first_seen TEXT,
            last_seen TEXT,
            confirmed_count INTEGER DEFAULT 0
        )
    ''')

    # Model evolution table — tracks model improvement over time
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            feedback_count INTEGER,
            true_positives INTEGER,
            false_positives INTEGER,
            true_negatives INTEGER,
            false_negatives INTEGER,
            precision REAL,
            recall REAL,
            f1 REAL,
            notes TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized.")

def save_scan(filename, results):
    conn = get_db()
    c = conn.cursor()

    timestamp = datetime.now().isoformat()

    c.execute('''
        INSERT INTO scans
        (filename, timestamp, total_sequences,
         attack_sequences, benign_sequences,
         attack_rate, avg_score, max_score, model_scores)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        timestamp,
        results['total_sequences'],
        results['attack_sequences'],
        results['benign_sequences'],
        results['attack_rate'],
        results['avg_ensemble_score'],
        results['max_ensemble_score'],
        json.dumps(results['model_scores'])
    ))

    scan_id = c.lastrowid

    # Save top detections
    for det in results.get('top_suspicious', []):
        c.execute('''
            INSERT INTO detections
            (scan_id, sequence_num, transformer_score,
             mlp_score, robust_score, ensemble_score,
             prediction, risk_level, attack_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_id,
            det['sequence'],
            det['transformer_score'],
            det['mlp_score'],
            det['robust_score'],
            det['ensemble_score'],
            det['prediction'],
            det['risk_level'],
            det.get('attack_type', 'UNKNOWN')
        ))

    conn.commit()
    conn.close()
    return scan_id

def save_feedback(detection_id, feedback):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        UPDATE detections
        SET feedback = ?,
            feedback_timestamp = ?
        WHERE id = ?
    ''', (feedback, datetime.now().isoformat(), detection_id))
    conn.commit()
    conn.close()

def get_scan_history(limit=50):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM scans
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_feedback_stats():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT
            feedback,
            COUNT(*) as count
        FROM detections
        WHERE feedback IS NOT NULL
        GROUP BY feedback
    ''')
    rows = c.fetchall()
    conn.close()
    stats = {r['feedback']: r['count'] for r in rows}
    return stats

def get_threat_trends(days=30):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT
            DATE(timestamp) as date,
            SUM(attack_sequences) as attacks,
            SUM(total_sequences) as total,
            AVG(attack_rate) as avg_rate
        FROM scans
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT ?
    ''', (days,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_threat_dna(dna_hash, attack_type, scores):
    conn = get_db()
    c = conn.cursor()
    now = datetime.now().isoformat()

    c.execute('''
        INSERT INTO threat_dna
        (dna_hash, attack_type, avg_transformer_score,
         avg_mlp_score, avg_ensemble_score,
         first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dna_hash) DO UPDATE SET
            occurrence_count = occurrence_count + 1,
            last_seen = ?
    ''', (
        dna_hash,
        attack_type,
        scores.get('transformer', 0),
        scores.get('mlp', 0),
        scores.get('ensemble', 0),
        now, now, now
    ))
    conn.commit()
    conn.close()

def get_threat_dna_library():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM threat_dna
        ORDER BY occurrence_count DESC
        LIMIT 100
    ''')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def find_dna_matches(dna_hash, threshold=0.8):
    """Find similar DNA signatures in the library"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM threat_dna
        WHERE dna_hash != ?
        ORDER BY occurrence_count DESC
        LIMIT 20
    ''', (dna_hash,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_dna_by_hash(dna_hash):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM threat_dna WHERE dna_hash = ?', (dna_hash,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def update_dna_confirmed(dna_hash):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        UPDATE threat_dna
        SET confirmed_count = confirmed_count + 1
        WHERE dna_hash = ?
    ''', (dna_hash,))
    conn.commit()
    conn.close()

def get_campaign_clusters():
    """Group DNA signatures by attack type"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT
            attack_type,
            COUNT(*) as signature_count,
            SUM(occurrence_count) as total_occurrences,
            AVG(avg_ensemble_score) as avg_score,
            MIN(first_seen) as first_seen,
            MAX(last_seen) as last_seen
        FROM threat_dna
        GROUP BY attack_type
        ORDER BY total_occurrences DESC
    ''')
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_dna_match(dna_hash, matched_hash, similarity_score):
    """Record when two DNA signatures match"""
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS dna_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dna_hash TEXT,
            matched_hash TEXT,
            similarity_score REAL,
            timestamp TEXT
        )
    ''')
    c.execute('''
        INSERT INTO dna_matches
        (dna_hash, matched_hash, similarity_score, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (dna_hash, matched_hash, similarity_score,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Initialize on import
init_db()