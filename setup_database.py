import sqlite3
import os
import json
from datetime import datetime, timedelta
import random
from werkzeug.security import generate_password_hash

# Database file
DB_FILE = 'enginsync.db'

# Check if database exists
db_exists = os.path.exists(DB_FILE)

# Connect to database
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

# Create tables if they don't exist
def create_tables():
    # Users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # UserSettings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UserSettings (
        settings_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        theme_preference TEXT DEFAULT 'system',
        notification_prefs TEXT,
        FOREIGN KEY (user_id) REFERENCES Users (user_id)
    )
    ''')
    
    # Courses table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Courses (
        course_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        category TEXT,
        difficulty_level TEXT,
        total_modules INTEGER DEFAULT 0
    )
    ''')
    
    # Skills table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Skills (
        skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT
    )
    ''')
    
    # UserProgress table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UserProgress (
        progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        course_id INTEGER NOT NULL,
        modules_completed INTEGER DEFAULT 0,
        last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completion_percentage REAL DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES Users (user_id),
        FOREIGN KEY (course_id) REFERENCES Courses (course_id)
    )
    ''')
    
    # UserSkills table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UserSkills (
        user_skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        skill_id INTEGER NOT NULL,
        proficiency_level REAL DEFAULT 0,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES Users (user_id),
        FOREIGN KEY (skill_id) REFERENCES Skills (skill_id)
    )
    ''')
    
    # UserGoals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UserGoals (
        goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        title TEXT NOT NULL,
        progress_percentage REAL DEFAULT 0,
        deadline TIMESTAMP,
        is_completed BOOLEAN DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES Users (user_id)
    )
    ''')
    
    # Activities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Activities (
        activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        activity_type TEXT NOT NULL,
        description TEXT NOT NULL,
        duration_minutes INTEGER,
        score REAL,
        status TEXT,
        completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES Users (user_id)
    )
    ''')

# Insert sample data
def insert_sample_data():
    # Sample user (only if no users exist)
    if not cursor.execute("SELECT COUNT(*) FROM Users").fetchone()[0]:
        cursor.execute(
            "INSERT INTO Users (email, password_hash, full_name) VALUES (?, ?, ?)",
            ("student@example.com", generate_password_hash("password"), "John Doe")
        )
        user_id = cursor.lastrowid
        
        # User settings
        cursor.execute(
            "INSERT INTO UserSettings (user_id, theme_preference, notification_prefs) VALUES (?, ?, ?)",
            (user_id, "system", json.dumps({"role": "student", "email_notifications": True}))
        )
    else:
        # Get existing user ID
        user_id = cursor.execute("SELECT user_id FROM Users LIMIT 1").fetchone()[0]
    
    # Sample courses
    courses = [
        ("Calculus I", "Introduction to calculus concepts", "Mathematics", "Intermediate", 10),
        ("Linear Algebra", "Fundamentals of linear algebra", "Mathematics", "Intermediate", 8),
        ("Thermodynamics", "Principles of thermodynamics", "Physics", "Advanced", 12),
        ("Statics", "Fundamentals of static mechanics", "Engineering", "Intermediate", 7),
        ("Programming Fundamentals", "Introduction to programming", "Computer Science", "Beginner", 15),
        ("Circuit Analysis", "Basic analysis of electrical circuits", "Electrical Engineering", "Intermediate", 9)
    ]
    
    # Only insert courses if none exist
    if not cursor.execute("SELECT COUNT(*) FROM Courses").fetchone()[0]:
        cursor.executemany(
            "INSERT INTO Courses (title, description, category, difficulty_level, total_modules) VALUES (?, ?, ?, ?, ?)",
            courses
        )
    
    # Get course IDs
    cursor.execute("SELECT course_id FROM Courses")
    course_ids = [row[0] for row in cursor.fetchall()]
    
    # Sample skills
    skills = [
        ("Calculus", "Mathematics"),
        ("Linear Algebra", "Mathematics"),
        ("Thermodynamics", "Physics"),
        ("Statics", "Engineering"),
        ("Programming", "Computer Science"),
        ("Circuits", "Electrical Engineering")
    ]
    
    # Only insert skills if none exist
    if not cursor.execute("SELECT COUNT(*) FROM Skills").fetchone()[0]:
        cursor.executemany(
            "INSERT INTO Skills (name, category) VALUES (?, ?)",
            skills
        )
    
    # Get skill IDs
    cursor.execute("SELECT skill_id FROM Skills")
    skill_ids = [row[0] for row in cursor.fetchall()]
    
    # User progress (only if none exists)
    if not cursor.execute("SELECT COUNT(*) FROM UserProgress").fetchone()[0]:
        for course_id in course_ids:
            # Random progress for each course
            completion = random.randint(30, 95)
            modules = cursor.execute("SELECT total_modules FROM Courses WHERE course_id = ?", (course_id,)).fetchone()[0]
            modules_completed = int(modules * completion / 100)
            
            cursor.execute(
                "INSERT INTO UserProgress (user_id, course_id, modules_completed, completion_percentage) VALUES (?, ?, ?, ?)",
                (user_id, course_id, modules_completed, completion)
            )
    
    # User skills (only if none exists)
    if not cursor.execute("SELECT COUNT(*) FROM UserSkills").fetchone()[0]:
        for skill_id in skill_ids:
            # Random proficiency for each skill
            proficiency = random.randint(40, 90)
            
            cursor.execute(
                "INSERT INTO UserSkills (user_id, skill_id, proficiency_level) VALUES (?, ?, ?)",
                (user_id, skill_id, proficiency)
            )
    
    # User goals (only if none exists)
    if not cursor.execute("SELECT COUNT(*) FROM UserGoals").fetchone()[0]:
        goals = [
            ("Complete Calculus Module 1", 75, datetime.now() + timedelta(days=10), 0),
            ("Read Physics Chapter 4", 90, datetime.now() + timedelta(days=5), 0),
            ("Project: Thermodynamics Sim", 40, datetime.now() + timedelta(days=14), 0)
        ]
        
        for goal in goals:
            cursor.execute(
                "INSERT INTO UserGoals (user_id, title, progress_percentage, deadline, is_completed) VALUES (?, ?, ?, ?, ?)",
                (user_id, goal[0], goal[1], goal[2], goal[3])
            )
    
    # User activities (only if none exists)
    if not cursor.execute("SELECT COUNT(*) FROM Activities").fetchone()[0]:
        now = datetime.now()
        activities = [
            ("assignment", "Calculus Practice Problems (Set 3)", 45, None, "completed", now - timedelta(days=3)),
            ("lecture", "Physics Lecture: Kinematics", 60, None, "watched", now - timedelta(days=4)),
            ("assignment", "Assignment 1: Mechanics", None, 85, "graded", now - timedelta(days=5)),
            ("lab", "Chemistry Lab Simulation", None, None, "pending", now - timedelta(days=6))
        ]
        
        for activity in activities:
            cursor.execute(
                "INSERT INTO Activities (user_id, activity_type, description, duration_minutes, score, status, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, activity[0], activity[1], activity[2], activity[3], activity[4], activity[5])
            )

# Execute database setup
create_tables()
insert_sample_data()

# Calculate overall progress for the user
def calculate_overall_progress():
    user_id = cursor.execute("SELECT user_id FROM Users LIMIT 1").fetchone()[0]
    
    # Get all user progress records
    cursor.execute("SELECT completion_percentage FROM UserProgress WHERE user_id = ?", (user_id,))
    progress_records = cursor.fetchall()
    
    if progress_records:
        # Calculate average completion percentage
        total_percentage = sum(record[0] for record in progress_records)
        overall_progress = total_percentage / len(progress_records)
        
        print(f"Overall progress calculated: {overall_progress:.2f}%")
    else:
        overall_progress = 0
        print("No progress records found.")

# Commit changes and close connection
conn.commit()
calculate_overall_progress()
conn.close()

print(f"Database setup completed successfully!")
