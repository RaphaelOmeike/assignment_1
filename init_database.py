"""
Database initialization script
Creates the SQLite database and tables for the emotion detection app.
"""

import sqlite3
import os

def init_database(db_path='database.db'):
    """
    Initialize the database with required tables.
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        # Remove existing database if it exists (for fresh start)
        if os.path.exists(db_path):
            print(f"Removing existing database: {db_path}")
            os.remove(db_path)
        
        # Create new database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Creating database tables...")
        
        # Create users table
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                age INTEGER NOT NULL,
                predicted_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_filename TEXT NOT NULL,
                image_data BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create an index on emotion for faster filtering
        cursor.execute('''
            CREATE INDEX idx_emotion ON users(predicted_emotion)
        ''')
        
        # Create an index on timestamp for faster sorting
        cursor.execute('''
            CREATE INDEX idx_timestamp ON users(timestamp)
        ''')
        
        # Commit changes
        conn.commit()
        
        print("Database initialized successfully!")
        print(f"Database file created: {db_path}")
        
        # Display table schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        
        print("\nTable schema:")
        print("Users table columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def add_sample_data(db_path='database.db'):
    """
    Add some sample data for testing (optional).
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Sample data
        sample_users = [
            ('John Doe', 'john@example.com', 25, 'happy', 0.95, 'sample1.jpg'),
            ('Jane Smith', 'jane@example.com', 30, 'sad', 0.87, 'sample2.jpg'),
            ('Bob Johnson', 'bob@example.com', 22, 'surprise', 0.76, 'sample3.jpg'),
            ('Alice Brown', 'alice@example.com', 28, 'neutral', 0.82, 'sample4.jpg'),
        ]
        
        print("Adding sample data...")
        
        for user in sample_users:
            cursor.execute('''
                INSERT INTO users (name, email, age, predicted_emotion, confidence, image_filename)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', user)
        
        conn.commit()
        conn.close()
        
        print(f"Added {len(sample_users)} sample users to the database")
        
    except Exception as e:
        print(f"Error adding sample data: {e}")

def view_database_contents(db_path='database.db'):
    """
    View the contents of the database.
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        if not os.path.exists(db_path):
            print(f"Database file {db_path} does not exist.")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get user count
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        
        print(f"\nDatabase contains {count} users")
        
        if count > 0:
            # Get recent users
            cursor.execute('''
                SELECT name, email, predicted_emotion, confidence, timestamp 
                FROM users 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''')
            
            recent_users = cursor.fetchall()
            
            print("\nRecent submissions:")
            print("-" * 80)
            print(f"{'Name':<20} {'Email':<25} {'Emotion':<10} {'Confidence':<12} {'Time'}")
            print("-" * 80)
            
            for user in recent_users:
                confidence_pct = f"{user[3]*100:.1f}%"
                print(f"{user[0]:<20} {user[1]:<25} {user[2]:<10} {confidence_pct:<12} {user[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error viewing database: {e}")

def main():
    """Main function to set up the database."""
    print("=== Database Initialization ===")
    
    # Initialize database
    success = init_database()
    
    if success:
        # Ask if user wants sample data
        response = input("\nWould you like to add sample data for testing? (y/n): ").lower()
        if response in ['y', 'yes']:
            add_sample_data()
        
        # View database contents
        view_database_contents()
        
        print("\n=== Database Setup Complete ===")
        print("The database is ready for the Flask application!")
        print("You can now run: python app.py")
    else:
        print("Database initialization failed!")

if __name__ == "__main__":
    main()