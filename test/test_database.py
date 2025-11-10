"""Test database connections"""
from config.database import MySQLDatabase, SQLiteDatabase

print("Testing Database Connections...")
print("=" * 60)

# Test MySQL
print("\n1. Testing MySQL Connection:")
mysql_db = MySQLDatabase()
if mysql_db.connect():
    print("   ✅ MySQL connected successfully!")
    
    # Test query
    result = mysql_db.fetch_one("SELECT COUNT(*) as count FROM workers")
    if result:
        print(f"   ✅ Found {result['count']} workers in database")
    
    mysql_db.close()
    print("   ✅ MySQL connection closed")
else:
    print("   ❌ MySQL connection failed")
    print("   Make sure XAMPP MySQL is running!")

# Test SQLite
print("\n2. Testing SQLite Database:")
sqlite_db = SQLiteDatabase()
print("   ✅ SQLite database initialized")

# Check if tables were created
import sqlite3
conn = sqlite3.connect(sqlite_db.db_path)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
conn.close()

print(f"   ✅ Created {len(tables)} tables:")
for table in tables:
    print(f"      - {table[0]}")

print("\n" + "=" * 60)
print("✅ All database tests completed!")