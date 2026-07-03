import sqlite3
from datetime import datetime
import os

# ------------------------------------
# Database Configuration
# ------------------------------------

DB_FOLDER = "database"
DB_NAME = "history.db"

os.makedirs(DB_FOLDER, exist_ok=True)

DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# ------------------------------------
# Connect Database
# ------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


# ------------------------------------
# Create Table
# ------------------------------------

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        prediction TEXT,
        confidence REAL,
        report TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()


# ------------------------------------
# Insert Prediction
# ------------------------------------

def save_prediction(
    image_name,
    prediction,
    confidence,
    report
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO prediction_history
    (
        image_name,
        prediction,
        confidence,
        report,
        created_at
    )
    VALUES (?, ?, ?, ?, ?)
    """,
    (
        image_name,
        prediction,
        confidence,
        report,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# ------------------------------------
# Get Prediction History
# ------------------------------------

def get_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT
        id,
        image_name,
        prediction,
        confidence,
        report,
        created_at
    FROM prediction_history
    ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


# ------------------------------------
# Delete History
# ------------------------------------

def delete_history():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM prediction_history")

    conn.commit()
    conn.close()


# ------------------------------------
# Initialize Database
# ------------------------------------

create_table()


# ------------------------------------
# Test
# ------------------------------------

if __name__ == "__main__":

    save_prediction(
        image_name="sample.jpg",
        prediction="PNEUMONIA",
        confidence=97.45,
        report="Possible pneumonia detected. Please consult a healthcare professional. This AI-generated report is for informational purposes only."
    )

    print("Prediction saved successfully.\n")

    history = get_history()

    for row in history:
        print(row)