import sqlite3

# 1. Connect to your database
conn = sqlite3.connect('IMDB.db')
cursor = conn.cursor()

# 2. Ask the database for a list of all its tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("--- 📂 TABLES FOUND IN DATABASE ---")
if not tables:
    print("Zero tables found! The database file might be empty.")
else:
    for t in tables:
        print(f"Table Name: {t[0]}")
        
conn.close()