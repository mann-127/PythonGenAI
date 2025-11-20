import sqlite3
import pandas as pd
 
# 1. Connect to SQLite database (creates file if not exists)
conn = sqlite3.connect("Database.db")
 
# 2. Create a cursor object
cursor = conn.cursor()
 
# 3. Create a table
cursor.execute("""
CREATE TABLE IF NOT EXISTS Employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary INTEGER
)
""")
 
# 4. Insert sample data
cursor.executemany("""
INSERT INTO Employees (name, department, salary) VALUES (?, ?, ?)
""", [
    ("Rahul", "Engineering", 70000),
    ("Sneha", "Engineering", 65000),
    ("Amit", "HR", 50000),
    ("Priya", "Finance", 60000),
    ("Vikram", "Engineering", 72000)
])
 
conn.commit()
 
# # 5. Query the data using SQL
cursor.execute("SELECT name, salary FROM Employees WHERE department='Engineering'")
rows = cursor.fetchall()
print("Engineering Employees:", rows)
 
# # 6. Use Pandas to load data directly from SQL
df = pd.read_sql_query("SELECT department, AVG(salary) as avg_salary FROM Employees GROUP BY department", conn)
print(df)
 
query1 = "SELECT name, salary FROM Employees WHERE department='Engineering' AND salary > 65000;"
print("Query 1 Results:", cursor.execute(query1).fetchall())
 
# 7. Close connection
conn.close()
