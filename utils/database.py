import sqlite3

def init_db():
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        sentiment TEXT,
        score REAL,
        file TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_result(text, sentiment, score, file):
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()

    c.execute("INSERT INTO results (text, sentiment, score, file) VALUES (?, ?, ?, ?)",
              (text, sentiment, score, file))

    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect("sentiment.db")
    c = conn.cursor()
    c.execute("SELECT * FROM results")
    data = c.fetchall()
    conn.close()
    return data