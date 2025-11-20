import sqlite3
import matplotlib.pyplot as plt
 
# Connect to database
conn = sqlite3.connect("Database.db")
cursor = conn.cursor()
 
# Fetch employee data
cursor.execute("SELECT name, salary FROM Employees")
rows = cursor.fetchall()
conn.close()
 
# Split into lists
names = [row[0] for row in rows]
salaries = [row[1] for row in rows]
 
# Plot bar chart
plt.figure(figsize=(8,5))
plt.bar(names, salaries, color='lightgreen')
plt.title("Employee Salaries from DB")
plt.xlabel("Employee")
plt.ylabel("Salary (INR)")
plt.show()
