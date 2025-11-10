"""Quick database connection test"""
from config.database import MySQLDatabase

print("Testing MySQL connection...")
db = MySQLDatabase()

if db.connect():
    print("✓ MySQL connection successful!")
    
    # Test query
    result = db.fetch_one("SELECT COUNT(*) as count FROM workers")
    print(f"✓ Found {result['count']} workers in database")
    
    db.close()
else:
    print("❌ MySQL connection failed!")
    print("Check your .env file and make sure MySQL is running")