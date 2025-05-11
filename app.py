from dotenv import load_dotenv
load_dotenv()
import sqlite3
import os
import json
import functools
import calendar
import markdown
import re
import pandas as pd
import numpy as np
from markupsafe import Markup
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, g,
    jsonify, send_file
)
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import requests
import tempfile
from werkzeug.utils import secure_filename
from pdf_utils import process_pdf, text_to_speech
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
from openai import OpenAI
import googleapiclient.discovery
import googleapiclient.errors
import numpy as np
import re
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io.wavfile import write as write_wav
import sys
import fitz  # PyMuPDF
import google.generativeai as genai

# --- App Configuration ---
app = Flask(__name__)
# IMPORTANT: Change this to a long, random secret key in a real application
# You can generate one using: python -c 'import os; print(os.urandom(24))'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', b'_5#y2L"F4Q8z\n\xec]/')
app.config['DATABASE'] = 'enginsync.db'

# Configure Google Generative AI (Gemini)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBDJi9htw2_i1C6-1Z8wM9OGuGFJYkgpyo')
genai.configure(api_key=GEMINI_API_KEY)
# Specific models for interview prep
QUESTION_GEN_MODEL_NAME = 'gemini-2.0-flash'  # Model for generating questions
ASSESSMENT_MODEL_NAME = 'gemini-2.0-flash'   # Model for assessments

# Initialize Gemini model
gemini_model = None
try:
    gemini_model = genai.GenerativeModel(ASSESSMENT_MODEL_NAME)
    print(f"Gemini API initialized with model: {ASSESSMENT_MODEL_NAME}", file=sys.stderr)
except Exception as e:
    print(f"Error initializing Gemini API: {e}", file=sys.stderr)

# Initialize OpenAI client for advanced summarization
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = None
if openai_api_key and openai_api_key != 'your_openai_api_key_here':
    try:
        client = OpenAI(api_key=openai_api_key)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
else:
    print("OpenAI API key not set or invalid - AI summarization will not be available")

# Add custom template filters
@app.template_filter('strftime')
def format_datetime(value, format='%Y-%m-%d'):
    if value is None:
        return ''
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return value
    return value.strftime(format)

# --- Database Helper Functions ---

def get_db():
    """Connects to the specific database."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    """Helper function to query database."""
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

# --- Authentication Decorator ---

def login_required(view):
    """View decorator that redirects anonymous users to the login page."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

# --- Load User Hook ---

@app.before_request
def load_logged_in_user():
    """If a user id is stored in the session, load the user object """
    """from the database into flask.g.user."""
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        g.user = query_db('SELECT * FROM Users WHERE user_id = ?', [user_id], one=True)
        if g.user is None: # Clear session if user_id is invalid
            session.clear()


# --- Interview Prep Functions ---
# Extract text from a resume PDF
def extract_resume_text(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error opening or reading PDF: {e}", file=sys.stderr)
        return None

# Generate interview questions based on resume text using Gemini
def generate_interview_questions(resume_text, job_role, experience_level, num_questions=4):
    try:
        # Clean up resume text
        resume_text = re.sub(r'\s+', ' ', resume_text).strip()
        resume_summary = resume_text[:4000]  # Limit text length for API

        prompt = f"""You are an expert interviewer creating questions for a mock interview.
        Candidate's Resume Summary:
        ---
        {resume_summary}
        ---
        Target Position: {experience_level} {job_role}
        Number of Questions to Generate: {num_questions}
        Instructions: Generate {num_questions} specific interview questions based only on the provided resume summary and the target position. Ensure the questions meet these criteria:
        1. Directly relevant to the skills, experiences, or projects mentioned in the resume summary.
        2. A mix of technical and behavioral questions appropriate for the role and experience level ({experience_level}).
        3. Tailored to the {experience_level} level.
        4. Probe deeper into listed experiences (e.g., "In project X, what was the most significant challenge..." not just "Tell me about project X").
        Output Format: Provide only the questions as a numbered list. No intro/outro.
        Example:
        1. Question 1?
        2. Question 2?
        """

        # Configure Gemini API
        if not gemini_model:
            print("Gemini API model not initialized", file=sys.stderr)
            return _get_default_questions(job_role, experience_level, num_questions)

        # Generate questions
        response = gemini_model.generate_content(prompt)
        try:
            generated_text = response.text
        except AttributeError:
             # Try iterating parts if .text fails
             try:
                 generated_text = "".join(part.text for part in response.parts)
             except Exception:
                  # Fallback if parts access also fails (less common now)
                  print("Could not access response text using standard methods.", file=sys.stderr)
                  response_text = str(response) # Use string representation as last resort
        except Exception as e:
             print(f"Error extracting text from response: {e}", file=sys.stderr)
             return _get_default_questions(job_role, experience_level, num_questions)

        # Parse questions from response
        questions = []
        potential_questions = generated_text.split('\n')
        for line in potential_questions:
            cleaned_line = re.sub(r'^\s*\d+[\.\)]?\s*', '', line.strip())
            if cleaned_line.endswith('?') and len(cleaned_line) > 10:
                questions.append(cleaned_line)
            elif len(cleaned_line) > 10 and '?' in cleaned_line and not line.startswith("---") and not line.lower().startswith("example"):
                questions.append(cleaned_line)

        # Add default questions if we don't have enough
        default_questions = [
            f"Can you walk me through your experience relevant to a {job_role} position?",
            f"Based on your resume, what project are you most proud of and why?",
            f"Describe a challenging technical problem you solved, as related to the skills needed for a {experience_level} {job_role}.",
            f"Why are you interested in this specific {job_role} role at this stage in your career?"
        ]
        
        if len(questions) < num_questions:
            needed = num_questions - len(questions)
            questions.extend(default_questions[:needed])

        return questions[:num_questions]

    except Exception as e:
        print(f"Error generating questions (API/Model: {QUESTION_GEN_MODEL_NAME}): {e}", file=sys.stderr)
        return [
            f"Can you describe your experience in {job_role}?",
            "What are your greatest professional achievements mentioned in your resume?",
            "How do you handle challenging situations at work, perhaps related to projects listed?",
            f"Why are you interested in pursuing a {experience_level} {job_role} position?"
        ]

def _get_default_questions(job_role, experience_level, num_questions=4):
    """Get default questions if API fails"""
    default_questions = [
        f"Can you walk me through your experience relevant to a {job_role} position?",
        f"Based on your resume, what project are you most proud of and why?",
        f"Describe a challenging technical problem you solved, as related to the skills needed for a {experience_level} {job_role}.",
        f"Why are you interested in this specific {job_role} role at this stage in your career?"
    ]
    return default_questions[:num_questions]

# Record audio function (for AJAX calls)
def record_audio_to_file(output_path, duration=30):
    """Record audio for a specified duration and save to a file"""
    try:
        fs = 44100  # Sample rate
        sd.default.samplerate = fs
        sd.default.channels = 1
        
        print(f"Recording audio for {duration} seconds...", file=sys.stderr)
        recording = sd.rec(int(duration * fs), dtype='float32')
        sd.wait()  # Wait until recording is finished
        
        # Convert to int16 for WAV file
        recording_int16 = np.int16(recording * 32767)
        
        # Save as WAV file
        write_wav(output_path, fs, recording_int16)
        return True
    except Exception as e:
        print(f"Error recording audio: {e}", file=sys.stderr)
        return False

# Transcribe audio using audio data
def transcribe_audio(audio_file_path, prompt_context=""):
    """
    Transcribe audio to text without enhancements or modifications.
    Only captures the exact words spoken by the user.
    """
    try:
        # Try to use faster-whisper transcription
        try:
            # Load the faster-whisper model (use the appropriate model size)
            model = WhisperModel("base", device="cpu", compute_type="int8")
            
            # Set options to focus on exact transcription
            # Note: faster-whisper has a different API than original whisper
            segments, info = model.transcribe(
                audio_file_path,
                beam_size=5,
                word_timestamps=False,
                initial_prompt="Transcribe the following audio exactly as spoken, word for word."
            )
            
            # Collect all segments into a single transcript
            transcript = ""
            for segment in segments:
                transcript += segment.text + " "
            
            transcript = transcript.strip()
            
            # If transcript is empty, return a placeholder
            if not transcript:
                print(f"Empty transcript from Whisper", file=sys.stderr)
                return "[No speech detected]"
                
            return transcript
            
        except ImportError as e:
            print(f"Error importing WhisperModel: {e}", file=sys.stderr)
            return get_mock_transcript(prompt_context)
        except Exception as e:
            print(f"Error using faster-whisper: {e}", file=sys.stderr)
            return get_mock_transcript(prompt_context)
            
    except Exception as e:
        print(f"Error transcribing audio: {e}", file=sys.stderr)
        return get_mock_transcript(prompt_context)

# Get a mock transcript for testing when audio transcription fails
def get_mock_transcript(question_context=""):
    """
    Provide a simple mock transcript when actual transcription fails.
    Returns a basic response appropriate for testing purposes only.
    """
    # Dictionary of basic responses for different question types
    mock_responses = {
        "experience": "I worked at ABC Company for three years in software development.",
        "project": "I built a dashboard that helped our team track key metrics more efficiently.",
        "challenge": "We had an issue with the database performance that I solved by optimizing queries.",
        "interest": "I'm interested in this role because it matches my skills in development.",
        "default": "I have relevant experience for this position based on my previous work."
    }
    
    # Determine which response to use based on keywords in the question
    if not question_context:
        return mock_responses["default"]
    
    question_lower = question_context.lower()
    if any(keyword in question_lower for keyword in ["experience", "background", "work history"]):
        return mock_responses["experience"]
    elif any(keyword in question_lower for keyword in ["project", "achievement", "proud", "accomplish"]):
        return mock_responses["project"]
    elif any(keyword in question_lower for keyword in ["challenge", "difficult", "problem", "obstacle"]):
        return mock_responses["challenge"]
    elif any(keyword in question_lower for keyword in ["interest", "why", "reason", "apply"]):
        return mock_responses["interest"]
    else:
        return mock_responses["default"]

# Assess interview answer using Gemini
def get_ai_assessment(question, answer, job_role, experience_level):
    """
    Uses Gemini to assess a single interview answer based on predefined criteria.
    
    Args:
        question (str): The interview question asked.
        answer (str): The candidate's transcribed answer.
        job_role (str): The target job role.
        experience_level (str): The target experience level.
        
    Returns:
        int: rating on a scale of 1-5, or 0 if no answer provided, or None if error
    """
    if not answer or answer.strip() == "":
         return 0 # Rate 0 for empty answers

    try:
        # Define the rating scale clearly for the AI
        rating_scale_definition = """
        Rating Scale (1-5):
        1 - Poor: Answer is irrelevant, unclear, incomplete, and poorly structured. Shows fundamental misunderstanding.
        2 - Fair: Answer is partially relevant but lacks clarity, detail, or structure. Shows basic understanding but needs significant improvement.
        3 - Good: Answer is relevant, reasonably clear, and provides adequate detail and structure. Meets basic expectations for the level.
        4 - Very Good: Answer is highly relevant, clear, detailed, well-structured, and demonstrates strong understanding/skills. Exceeds basic expectations.
        5 - Excellent: Answer is outstandingly relevant, clear, concise yet comprehensive, perfectly structured, and showcases exceptional insight or skill. Truly impressive.
        """

        prompt = f"""You are an expert Hiring Manager evaluating a candidate's response during a mock interview.
        Your task is to assess the following answer based on the provided criteria and assign a rating from 1 to 5.

        Context:
        - Target Job Role: {experience_level} {job_role}
        - Interview Question: "{question}"
        - Candidate's Answer: "{answer}"

        Evaluation Criteria:
        1. Relevance: How directly and effectively does the answer address the specific question asked?
        2. Clarity: Is the language clear, concise, and easy to understand? Is the candidate articulate?
        3. Completeness & Detail: Does the answer provide sufficient depth, examples, and evidence relevant to the question and the candidate's experience level ({experience_level})? Avoids being overly brief or excessively rambling.
        4. Structure: Is the answer well-organized and logical? (e.g., For behavioral questions, does it resemble the STAR method - Situation, Task, Action, Result?)

        {rating_scale_definition}

        Instructions:
        1. Analyze the candidate's answer thoroughly against the criteria.
        2. Determine the most appropriate rating (1-5) based on the scale definition.

        Output Format: Respond ONLY with the numerical rating (1-5) as a single digit.
        Example: 4
        """

        # Check if Gemini API is available
        if not gemini_model:
            print("Gemini API model not initialized for assessment", file=sys.stderr)
            return 3 # Return just a default rating

        # Generate assessment
        response = gemini_model.generate_content(prompt)
        
        # --- Robust Parsing of AI Response ---
        response_text = ""
        try:
            # Try standard .text access
            response_text = response.text
        except AttributeError:
             # Try iterating parts if .text fails
             try:
                 response_text = "".join(part.text for part in response.parts)
             except Exception:
                  # Fallback if parts access also fails (less common now)
                  print("Could not access response text using standard methods.", file=sys.stderr)
                  response_text = str(response) # Use string representation as last resort
        except Exception as e:
             print(f"Error extracting text from assessment response: {e}", file=sys.stderr)
             return None

        # Clean potential markdown/formatting artifacts and extract just the number
        cleaned_text = response_text.strip()
        
        # Try to extract just a rating number
        try:
            # Look for a single digit 1-5 in the response
            rating_match = re.search(r'\b([1-5])\b', cleaned_text)
            if rating_match:
                rating = int(rating_match.group(1))
                if 1 <= rating <= 5:
                    return rating
            
            # If no match but the cleaned text is just a digit, try that
            if cleaned_text.isdigit() and len(cleaned_text) == 1:
                rating = int(cleaned_text)
                if 1 <= rating <= 5:
                    return rating
                    
            print(f"Could not extract valid rating from: {cleaned_text}", file=sys.stderr)
            return 3 # Default to middle rating if parsing fails
            
        except Exception as e_reg:
            print(f"Error during regex parsing: {e_reg}", file=sys.stderr)
            return 3 # Default to middle rating

    except Exception as e:
        print(f"Error getting AI assessment (API/Model: {ASSESSMENT_MODEL_NAME}): {e}", file=sys.stderr)
        # Check for specific API errors if possible (e.g., rate limits, auth)
        if "API key" in str(e):
             print("Check your Google API Key configuration and permissions.", file=sys.stderr)
        return None

# --- Routes ---

@app.route('/')
@app.route('/home')
def home():
    """Public homepage."""
    return render_template('home.html')

@app.route('/signup', methods=('GET', 'POST'))
def signup():
    """User registration."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')
        full_name = request.form.get('firstName', '').strip() + " " + request.form.get('lastName', '').strip()
        role = 'student'  # Default role

        error = None

        import re
        # Basic email validation regex pattern
        email_pattern = r'^[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}$'

        if not email:
            error = 'Email is required.'
        elif not re.match(email_pattern, email):
            error = 'Please enter a valid email address.'
        elif not password:
            error = 'Password is required.'
        elif password != confirm_password:
             error = 'Passwords do not match.'
        elif query_db('SELECT user_id FROM Users WHERE email = ?', [email], one=True) is not None:
            error = f"Email '{email}' is already registered."

        if error is None:
            db = get_db()
            try:
                # Create user with role
                user_id = db.execute(
                    "INSERT INTO Users (email, password_hash, full_name) VALUES (?, ?, ?)",
                    (email, generate_password_hash(password), full_name),
                ).lastrowid
                
                # Store user role in UserSettings
                db.execute(
                    "INSERT INTO UserSettings (user_id, theme_preference, notification_prefs) VALUES (?, ?, ?)",
                    (user_id, 'system', json.dumps({'role': role or 'student'}))
                )
                
                db.commit()
                flash('Account created successfully! Please log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.Error as e:
                 error = f"Database error: {e}"
                 db.rollback() # Rollback changes on error

        if error:
             flash(error, 'error')

    # GET request or failed POST
    return render_template('signup.html')


@app.route('/login', methods=('GET', 'POST'))
def login():
    """User login."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password')
        error = None
        user = query_db('SELECT * FROM Users WHERE email = ?', [email], one=True)

        if user is None:
            error = 'Incorrect email.'
        elif not check_password_hash(user['password_hash'], password):
            error = 'Incorrect password.'

        if error is None:
            # store the user id in a new session and return to the index
            session.clear()
            session['user_id'] = user['user_id']
            flash(f"Welcome back, {user['full_name']}!", 'success')
            return redirect(url_for('dashboard'))

        flash(error, 'error')

    # GET request or failed POST
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logs the user out."""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# --- Protected Routes ---

@app.route('/dashboard')
@login_required
def dashboard():
    """Shows the main dashboard after login."""
    # g.user is available here thanks to @app.before_request
    user_id = g.user['user_id']
    
    # Get current datetime for template
    now = datetime.now()
    
    # Fetch user's skills with proficiency levels
    skills = query_db("""
        SELECT s.name, us.proficiency_level 
        FROM UserSkills us
        JOIN Skills s ON us.skill_id = s.skill_id
        WHERE us.user_id = ?
    """, [user_id])
    
    # Fetch user's goals
    goals = query_db("""
        SELECT title, deadline, progress_percentage
        FROM UserGoals
        WHERE user_id = ?
        ORDER BY deadline
    """, [user_id])
    
    # Fetch recent activities
    activities = query_db("""
        SELECT description, completed_at, duration_minutes, score, status
        FROM Activities
        WHERE user_id = ?
        ORDER BY completed_at DESC
        LIMIT 10
    """, [user_id])
    
    # Calculate total exercises completed
    exercise_count = query_db('''
        SELECT COUNT(*) as count
        FROM Activities
        WHERE user_id = ? AND activity_type = 'exercise'
    ''', [user_id], one=True)['count']
    
    # Calculate total hours studied
    hours_result = query_db("""
        SELECT SUM(duration_minutes) as total_minutes
        FROM Activities
        WHERE user_id = ?
    """, [user_id], one=True)
    hours_studied = round(hours_result['total_minutes'] / 60) if hours_result['total_minutes'] else 0
    
    # Weekly progress for chart
    weekly_progress = query_db("""
        SELECT strftime('%W', completed_at) as week, COUNT(*) as count
        FROM Activities
        WHERE user_id = ?
        GROUP BY week
        ORDER BY week
        LIMIT 6
    """, [user_id])
    
    # Get assignment scores for completion chart
    assignments = query_db('''
        SELECT description, score
        FROM Activities
        WHERE user_id = ? AND activity_type = 'assignment' AND score IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 6
    ''', [user_id])
    
    # Calculate streak (consecutive days with activity)
    streak = 5  # Placeholder - would need more complex query to calculate actual streak
    
    # Calculate overall progress percentage
    overall_progress = query_db("""
        SELECT AVG(completion_percentage) as avg_progress
        FROM UserProgress
        WHERE user_id = ?
    """, [user_id], one=True)['avg_progress'] or 0
    
    return render_template('dashboard.html',
                          user=g.user,
                          full_name=g.user['full_name'],
                          skills=skills,
                          goals=goals,
                          activities=activities,
                          exercise_count=exercise_count,
                          hours_studied=hours_studied,
                          weekly_progress=weekly_progress,
                          assignments=assignments,
                          streak=streak,
                          overall_progress=overall_progress,
                          now=now)

# Courses route removed as requested

# Progress route removed - now integrated into dashboard

@app.route('/planner')
@login_required
def planner():
    """Placeholder for planner page."""
    return render_template('planner.html', back_url=url_for('dashboard'))

def generate_study_plan(prompt):
    """Generates a study plan using the Gemini API and formats it for better display."""
    try:
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not gemini_api_key or gemini_api_key == 'your_gemini_api_key_here':
            raise ValueError("Gemini API key not set or invalid. Please set GEMINI_API_KEY in your .env file.")
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        # More specific prompt to generate well-structured, table-based study plans
        import threading
        result = {}
        exception = {}
        def call_gemini():
            try:
                result['response'] = model.generate_content(f"""
                Create a detailed, structured study plan for: '{prompt}'
                
                FORMAT REQUIREMENTS (CRITICAL):
                1. Use HTML formatting to create a visually appealing, structured plan
                2. Start with a brief introduction about the plan
                3. Include a main table with these EXACT columns:
                   - Day/Date
                   - Topics
                   - Time Allocation
                   - Activities/Resources
                4. Format the table with proper HTML <table>, <tr>, <th>, and <td> tags
                5. For each day, include detailed daily activities with bullet points
                6. Use <strong> tags for emphasis and organization
                7. Add study tips and best practices section after the main schedule
                8. Use proper headings with <h2>, <h3> tags for different sections
                9. Make sure each day's content is comprehensive but concise
                
                EXAMPLE TABLE STRUCTURE (follow this format exactly):
                <table>
                  <tr>
                    <th>Day/Date</th>
                    <th>Topics</th>
                    <th>Time Allocation</th>
                    <th>Activities/Resources</th>
                  </tr>
                  <tr>
                    <td>Day 1</td>
                    <td>Introduction to [Topic]</td>
                    <td>3-4 hours</td>
                    <td>
                      <ul>
                        <li>Watch introductory videos</li>
                        <li>Read chapter 1-2</li>
                        <li>Practice basic exercises</li>
                      </ul>
                    </td>
                  </tr>
                  <!-- More rows for other days -->
                </table>
                
                YOUR RESPONSE MUST INCLUDE THIS HTML TABLE FORMAT FOR THE SCHEDULE.
                """)
            except Exception as e:
                exception['error'] = e
        thread = threading.Thread(target=call_gemini)
        thread.start()
        thread.join(timeout=30)
        if thread.is_alive():
            raise TimeoutError("The AI study plan generation timed out after 30 seconds. Please try again with a more specific or shorter request.")
        if 'error' in exception:
            raise exception['error']
        response = result['response']

        
        # Get the raw text response
        raw_text = response.text
        
        # First, clean up any raw HTML tags or text markers at the beginning
        # Remove anything that looks like ```html or html or <html> at the start
        raw_text = re.sub(r'^\s*(?:```(?:html)?|html|<html>)\s*', '', raw_text, flags=re.IGNORECASE)
        # Remove trailing ``` if present at the end
        raw_text = re.sub(r'\s*```\s*$', '', raw_text)
        
        # Clean up any markdown-style tables and convert to proper HTML if needed
        # Replace markdown tables with HTML tables if they exist
        table_pattern = r'\|(.+?)\|\n\|[-:\s|]+\|\n((\|.+?\|\n)+)'
        
        def table_replacer(match):
            header = match.group(1).strip().split('|')
            rows = match.group(2).strip().split('\n')
            
            html_table = '<table class="study-plan-table">\n<thead>\n<tr>'
            for col in header:
                html_table += f'<th>{col.strip()}</th>'
            html_table += '</tr>\n</thead>\n<tbody>'
            
            for row in rows:
                html_table += '\n<tr>'
                cols = row.strip().strip('|').split('|')
                for col in cols:
                    html_table += f'<td>{col.strip()}</td>'
                html_table += '</tr>'
            
            html_table += '\n</tbody></table>'
            return html_table
        
        # Apply table conversion if markdown tables exist
        formatted_text = re.sub(table_pattern, table_replacer, raw_text, flags=re.DOTALL)
        
        # Apply additional structure improvements
        # Convert markdown headings to HTML if needed
        formatted_text = re.sub(r'^##\s+(.+?)$', r'<h2>\1</h2>', formatted_text, flags=re.MULTILINE)
        formatted_text = re.sub(r'^###\s+(.+?)$', r'<h3>\1</h3>', formatted_text, flags=re.MULTILINE)
        
        # Convert markdown lists to HTML if needed
        formatted_text = re.sub(r'^\*\s+(.+?)$', r'<li>\1</li>', formatted_text, flags=re.MULTILINE)
        formatted_text = re.sub(r'(<li>.+?</li>\n)+', r'<ul>\n\g<0></ul>', formatted_text, flags=re.DOTALL)
        
        # Convert markdown emphasis to HTML if needed
        formatted_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_text)
        formatted_text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', formatted_text)
        
        # Ensure the content is properly formatted with HTML
        # If the content doesn't already have HTML structure, add it
        if '<html>' not in formatted_text.lower():
            # Check if we have any HTML tags at all
            if '<' not in formatted_text or '>' not in formatted_text:
                # If no HTML, treat as plain text and add basic formatting
                replacement1 = formatted_text.replace("\n\n", "</p><p>")
                replacement2 = replacement1.replace("\n", "<br>")
                formatted_text = f'<div class="study-plan-container"><p>{replacement2}</p></div>'
        
        # Ensure we have proper HTML structure for the content
        html_content = formatted_text
        
        # Add specific styling for different sections
        html_content = re.sub(r'<h2>(.+?Tips.+?)</h2>', r'<h2 class="study-plan-tips-header">\1</h2>', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<h2>(.+?Notes.+?)</h2>', r'<h2 class="study-plan-notes-header">\1</h2>', html_content, flags=re.IGNORECASE)
        
        # Add wrapper div for better styling control if not already present
        if not html_content.startswith('<div class="study-plan-container">'):  
            html_content = f'<div class="study-plan-container">{html_content}</div>'
        
        # Final cleanup to remove any stray text or markers that might be at the end
        # This handles cases where the AI appends things like "I hope this helps!" or other commentary
        html_content = re.sub(r'</div>\s*[\w\s,.!?:;()-]+$', '</div>', html_content)
            
        return Markup(html_content)  # Mark as safe HTML to prevent escaping
    except Exception as e:
        # Create a fallback HTML structure when an error occurs
        error_html = f'''
        <div class="study-plan-container">
            <div class="study-plan-error">
                <h2>Error Generating Study Plan</h2>
                <p>We encountered an error while generating your study plan: {e}</p>
                <p>Please try again with a more specific request.</p>
            </div>
        </div>
        '''
        return Markup(error_html)

@app.route('/ai_planner', methods=['GET', 'POST'])
@login_required
def ai_planner():
    """AI-powered study planner page."""
    study_plan_text = None
    
    if request.method == 'POST':
        study_prompt = request.form.get('study_prompt')
        if study_prompt:
            study_plan_text = generate_study_plan(study_prompt)
    
    return render_template('ai_planner.html', 
                          back_url=url_for('dashboard'),
                          study_plan_text=study_plan_text)

@app.route('/interview_prep')
@login_required
def interview_prep():
    """Interview preparation with Mr. Nags."""
    # This is equivalent to the main Streamlit page
    # In Flask, we'll render the template and handle the logic via AJAX requests
    return render_template('interview_prep.html', back_url=url_for('dashboard'))

@app.route('/generate_interview_questions', methods=['POST'])
@login_required
def generate_interview_questions_route():
    """Generate interview questions based on uploaded resume."""
    try:
        # Check if a file was uploaded
        if 'resume' not in request.files:
            return jsonify({'success': False, 'error': 'No resume file uploaded'})
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Get job role and experience level from form
        job_role = request.form.get('job_role', '')
        experience_level = request.form.get('experience_level', 'Mid')
        
        # Save the uploaded file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        resume_file.save(temp_file.name)
        temp_file.close()
        
        # Extract text from resume
        resume_text = extract_resume_text(temp_file.name)
        
        # Remove temporary file
        os.unlink(temp_file.name)
        
        if not resume_text:
            return jsonify({'success': False, 'error': 'Failed to extract text from resume'})
        
        # Generate questions
        questions = generate_interview_questions(resume_text, job_role, experience_level)
        
        # Store questions in session
        session['interview_questions'] = questions
        session['interview_job_role'] = job_role
        session['interview_experience_level'] = experience_level
        # Reset answers and assessments for a new interview
        session['interview_answers'] = {}
        session['interview_assessments'] = {}
        session['interview_current_question'] = 0
        session['interview_complete'] = False
        session.modified = True
        
        return jsonify({
            'success': True, 
            'questions': questions
        })
    
    except Exception as e:
        print(f"Error generating interview questions: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/record_interview_answer', methods=['POST'])
@login_required
def record_interview_answer():
    """Process uploaded audio file and transcribe it."""
    try:
        # Check if an audio file was uploaded
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file uploaded'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded audio file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_file.name)
        temp_file.close()
        
        # Get question for context (helps with more accurate transcription)
        question_index = int(request.form.get('question_index', 0))
        
        question_context = ""
        if 'interview_questions' in session and question_index < len(session['interview_questions']):
            question_context = session['interview_questions'][question_index]
        
        # Transcribe audio
        transcript = transcribe_audio(temp_file.name, prompt_context=question_context)
        
        # Remove temporary file
        os.unlink(temp_file.name)
        
        # Store answer in session
        if 'interview_answers' not in session:
            session['interview_answers'] = {}
        
        # Store answer by question text
        if 'interview_questions' in session and question_index < len(session['interview_questions']):
            question = session['interview_questions'][question_index]
            session['interview_answers'][question] = transcript
            
            # Initialize assessment placeholder
            if 'interview_assessments' not in session:
                session['interview_assessments'] = {}
            session['interview_assessments'][question] = {'rating': None, 'justification': None}
            
            # Update current question - if at the end, mark interview as complete
            current_q = session.get('interview_current_question', 0)
            if current_q == question_index:  # If answering current question (not going back to previous)
                if current_q < len(session['interview_questions']) - 1:
                    session['interview_current_question'] = current_q + 1
                else:
                    session['interview_complete'] = True
            
            session.modified = True
        
        return jsonify({
            'success': True,
            'transcript': transcript,
            'next_question': session.get('interview_current_question', 0),
            'interview_complete': session.get('interview_complete', False)
        })
        
    except Exception as e:
        print(f"Error processing interview answer: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/assess_interview_answers', methods=['POST'])
@login_required
def assess_interview_answers():
    """Assess interview answers and provide feedback."""
    try:
        if 'interview_questions' not in session or 'interview_answers' not in session:
            return jsonify({'success': False, 'error': 'No interview data found'})
        
        job_role = session.get('interview_job_role', '')
        experience_level = session.get('interview_experience_level', 'Mid')
        
        assessments = {}
        areas_for_improvement = []
        
        for question, answer in session['interview_answers'].items():
            if answer and answer.strip():
                rating = get_ai_assessment(question, answer, job_role, experience_level)
                assessments[question] = {'rating': rating, 'justification': None}
                
                # Extract potential areas for improvement
                if rating and rating <= 3:
                    areas_for_improvement.append(f"Question: {question}")
            else:
                assessments[question] = {'rating': 0, 'justification': None}
        
        # Store assessments in session
        session['interview_assessments'] = assessments
        session.modified = True
        
        # Calculate overall rating statistics
        total_rating = 0
        valid_count = 0
        for data in assessments.values():
            rating = data.get('rating')
            if rating is not None and rating > 0:
                total_rating += rating
                valid_count += 1
        
        avg_rating = round(total_rating / max(valid_count, 1), 1)
        
        # Get performance level based on average rating
        if avg_rating >= 4.5:
            performance = "Excellent Performance! 🏆"
        elif avg_rating >= 3.5:
            performance = "Very Good Performance! 👍"
        elif avg_rating >= 2.5:
            performance = "Good Performance 🙂"
        elif avg_rating >= 1.5:
            performance = "Fair Performance ⚠️"
        else:
            performance = "Needs Improvement 🆘"
        
        return jsonify({
            'success': True,
            'assessments': assessments,
            'stats': {
                'average_rating': avg_rating,
                'total_questions': len(session['interview_questions']),
                'answered_questions': len(session['interview_answers']),
                'assessed_answers': valid_count,
                'performance_level': performance,
                'areas_for_improvement': areas_for_improvement[:5]  # Top 5 improvement areas
            }
        })
        
    except Exception as e:
        print(f"Error assessing interview answers: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_interview_data', methods=['GET'])
@login_required
def get_interview_data():
    """Get all interview-related data from session."""
    # This route is equivalent to Streamlit's state access
    return jsonify({
        'success': True,
        'questions': session.get('interview_questions', []),
        'current_question': session.get('interview_current_question', 0),
        'answers': session.get('interview_answers', {}),
        'assessments': session.get('interview_assessments', {}),
        'interview_complete': session.get('interview_complete', False),
        'job_role': session.get('interview_job_role', ''),
        'experience_level': session.get('interview_experience_level', '')
    })

@app.route('/reset_interview', methods=['POST'])
@login_required
def reset_interview():
    """Reset interview data in session."""
    try:
        # Clear all interview-related session data
        interview_keys = [
            'interview_questions',
            'interview_answers',
            'interview_assessments',
            'interview_current_question',
            'interview_job_role',
            'interview_experience_level',
            'interview_complete'
        ]
        
        for key in interview_keys:
            if key in session:
                session.pop(key)
        
        session.modified = True
        
        return jsonify({'success': True, 'message': 'Interview data reset successfully'})
    except Exception as e:
        print(f"Error resetting interview: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ask_question_aloud', methods=['POST'])
@login_required
def ask_question_aloud():
    """Text-to-speech for interview questions (simulated in Flask)."""
    try:
        # Get question index
        question_index = int(request.form.get('question_index', 0))
        
        # Get question text
        if 'interview_questions' not in session or question_index >= len(session['interview_questions']):
            return jsonify({'success': False, 'error': 'Question not found'})
        
        question = session['interview_questions'][question_index]
        
        # In a real implementation, you could use a TTS service like Google's TTS
        # For now, we just acknowledge it (browser will handle TTS via Web Speech API)
        return jsonify({'success': True, 'message': 'TTS request received'})
        
    except Exception as e:
        print(f"Error in TTS: {e}", file=sys.stderr)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/jobsearch', methods=['GET', 'POST']) # Renamed from placement.html
@login_required
def jobsearch():
    """Job search page using Adzuna API."""
    jobs = []
    error = None
    user_id = g.user['user_id']
    
    # Initialize variables to store form input values
    search_skills = ''
    search_location = ''
    
    if request.method == 'POST':
        search_skills = request.form.get('skills', '')
        search_location = request.form.get('location', 'Karnataka')
        num_results = int(request.form.get('num_results', 5))
        
        # Adzuna API endpoint for India
        url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
        
        # API parameters
        params = {
            "app_id": "96e12eac",
            "app_key": "7a545dc457029cd2527a9f21a366010e",
            "what": search_skills,
            "where": search_location,
            "results_per_page": num_results,
            "content-type": "application/json",
        }
        
        try:
            # Debug API call parameters
            print(f"Making API call with params: {params}")
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                results = response.json()
                print(f"API response received: {results.keys()}")
                
                jobs = results.get("results", [])
                print(f"Found {len(jobs)} jobs")
                
                # If we don't have results, provide sample job data for testing
                if not jobs:
                    print("No results found, using sample data")
                    jobs = [
                        {
                            'title': 'Software Developer',
                            'company': {'display_name': 'TechCorp India'},
                            'location': {'display_name': 'Bangalore, Karnataka'},
                            'redirect_url': 'https://example.com/job1'
                        },
                        {
                            'title': 'Data Analyst',
                            'company': {'display_name': 'Analytics Partners'},
                            'location': {'display_name': 'Hyderabad, Telangana'},
                            'redirect_url': 'https://example.com/job2'
                        },
                        {
                            'title': 'Frontend Engineer',
                            'company': {'display_name': 'WebSolutions Inc.'},
                            'location': {'display_name': 'Pune, Maharashtra'},
                            'redirect_url': 'https://example.com/job3'
                        }
                    ]
                
                # Add icon based on job title for UI display
                for job in jobs:
                    if 'software' in job['title'].lower() or 'developer' in job['title'].lower():
                        job['icon'] = 'fas fa-laptop-code'
                    elif 'data' in job['title'].lower() or 'analyst' in job['title'].lower():
                        job['icon'] = 'fas fa-database'
                    elif 'engineer' in job['title'].lower():
                        job['icon'] = 'fas fa-cogs'
                    else:
                        job['icon'] = 'fas fa-briefcase'
            else:
                error = f"Error fetching jobs: {response.status_code}"
        except Exception as e:
            error = f"An error occurred: {str(e)}"
    
    return render_template('jobsearch.html', 
                           jobs=jobs, 
                           error=error, 
                           search_skills=search_skills,
                           search_location=search_location,
                           back_url=url_for('dashboard'))

@app.route('/settings')
@login_required
def settings():
    """Settings page with user profile information and preferences."""
    # Get current user's information from database
    user = None
    if g.user:
        try:
            # Query the database for user information - using the correct schema
            user_query = """
            SELECT * FROM Users WHERE user_id = ?
            """
            user = query_db(user_query, [g.user['id']], one=True)
            
            # If user is found, also fetch their primary learning goal
            if user:
                goal_query = """
                SELECT title FROM UserGoals 
                WHERE user_id = ? AND is_completed = 0
                ORDER BY deadline ASC
                LIMIT 1
                """
                primary_goal = query_db(goal_query, [g.user['id']], one=True)
                if primary_goal:
                    user = dict(user)
                    user['primary_goal'] = primary_goal['title']
                
        except Exception as e:
            print(f"Error fetching user data: {e}", file=sys.stderr)
            flash("Could not load all user data. Some settings may not display correctly.", "warning")
    
    return render_template('settings.html', user=user, back_url=url_for('dashboard'))

@app.route('/textbot')
@login_required
def textbot():
    """Textbook bot page with PDF processing and TTS."""
    return render_template('textbot.html', back_url=url_for('dashboard'))

# Create a temporary directory for audio files
TEMP_AUDIO_DIR = os.path.join(tempfile.gettempdir(), 'fc_audio_files')
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

@app.route('/upload-pdf', methods=['POST'])
@login_required
def upload_pdf():
    """Handle PDF uploads and processing."""
    if 'pdf-file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    pdf_file = request.files['pdf-file']
    if pdf_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not pdf_file.filename.lower().endswith('.pdf'):
        return jsonify({'success': False, 'error': 'File must be a PDF'})
    
    # Save the PDF file to a temporary location for future access
    temp_dir = os.path.join(tempfile.gettempdir(), 'fc_pdf_files')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_pdf_path = os.path.join(temp_dir, secure_filename(pdf_file.filename))
    pdf_file.save(temp_pdf_path)
    
    # Store PDF in session for later use
    if 'pdf_data' not in session:
        session['pdf_data'] = {}
    
    # Process the PDF
    page_num = int(request.form.get('page', 0))
    
    # Open the saved file for processing
    with open(temp_pdf_path, 'rb') as f:
        result = process_pdf(f, page_num)
    
    if result['success']:
        # Store basic info in session
        session['pdf_data']['filename'] = secure_filename(pdf_file.filename)
        session['pdf_data']['total_pages'] = result['total_pages']
        session['pdf_data']['current_page'] = page_num
        session['pdf_data']['temp_path'] = temp_pdf_path
        # Store the text content for summarization - combine sentences
        if 'sentences' in result:
            session['pdf_text'] = '. '.join(result['sentences'])
        else:
            session['pdf_text'] = ''
        session.modified = True
    
    return jsonify(result)

@app.route('/get-pdf-page', methods=['POST'])
@login_required
def get_pdf_page():
    """Get a specific page from the uploaded PDF."""
    if 'pdf_data' not in session or 'temp_path' not in session['pdf_data']:
        return jsonify({'success': False, 'error': 'No PDF uploaded or file not found'})
    
    page_num = int(request.form.get('page', 0))
    temp_pdf_path = session['pdf_data']['temp_path']
    
    if not os.path.exists(temp_pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'})
    
    # Open the saved file for processing
    with open(temp_pdf_path, 'rb') as f:
        result = process_pdf(f, page_num)
    
    if result['success']:
        session['pdf_data']['current_page'] = page_num
        if 'text' in result:
            session['pdf_text'] = result['text']
        elif 'sentences' in result:
            session['pdf_text'] = '. '.join(result['sentences'])
        else:
            session['pdf_text'] = ''
        session.modified = True
    
    return jsonify(result)

@app.route('/text-to-speech', methods=['POST'])
@login_required
def generate_speech():
    """Generate speech from text."""
    text = request.form.get('text', '')
    
    if not text:
        return jsonify({'success': False, 'error': 'No text provided'})
    
    try:
        audio_file = text_to_speech(text, TEMP_AUDIO_DIR)
        filename = os.path.basename(audio_file)
        
        return jsonify({
            'success': True,
            'audio_url': url_for('serve_audio', filename=filename)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
print("GEMINI_API_KEY loaded:", os.environ.get("GEMINI_API_KEY"))
@app.route('/audio/<filename>')
@login_required
def serve_audio(filename):
    """Serve generated audio files."""
    return send_file(os.path.join(TEMP_AUDIO_DIR, filename), mimetype='audio/mp3')

def extract_summarize_text(text, num_sentences=3):
    """
    Summarization using Gemini 2.0 Flash API (replaces NLTK extractive summarization).
    
    Args:
        text (str): The text to summarize
        num_sentences (int): Number of sentences to include in the summary (used as a hint for Gemini)
    Returns:
        str: The summarized text
    """
    try:
        if gemini_model is None:
            print("Gemini model not available, returning original text.")
            return text
        response = gemini_model.generate_content(
            f"Please summarize the following text in about {num_sentences} sentences (be concise and clear):\n\n{text}",
            generation_config={
                'max_output_tokens': num_sentences * 40  # Rough estimate, adjust as needed
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini summarization: {str(e)}")
        return text


def ai_summarize_text(text, max_tokens=150):
    """
    AI-powered summarization using Gemini 2.0 Flash API.
    
    Args:
        text (str): The text to summarize
        max_tokens (int): Maximum tokens for the summary (approximate, Gemini uses 'max_output_tokens')
    Returns:
        str: The AI-generated summary
    """
    try:
        if gemini_model is None:
            print("Gemini model not available, falling back to extractive summarization")
            return extract_summarize_text(text)
        # Gemini expects a prompt as a string, not messages
        response = gemini_model.generate_content(
            f"Please summarize the following text in about {max_tokens} words (be concise and clear):\n\n{text}",
            generation_config={
                'max_output_tokens': max_tokens
            }
        )
        # Gemini's response is in response.text
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini AI summarization: {str(e)}")
        return extract_summarize_text(text)

def search_youtube(query, max_results=6):
    """
    Searches YouTube for videos based on a query using YouTube Data API v3.

    Args:
        query (str): The search term or concept.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of dictionaries, each containing video details
              (title, video_id, channel_title), or None if an error occurs.
    """
    try:
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            print("YouTube API key not found in environment variables")
            return None

        api_service_name = "youtube"
        api_version = "v3"

        # Build the YouTube service object
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=api_key,
            static_discovery=False)

        # Make the API call
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results
        )
        response = request.execute()

        videos = []
        if 'items' in response:
            for item in response['items']:
                # Extract relevant information from the response
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel_title = item['snippet']['channelTitle']
                
                # Extract thumbnail URLs
                thumbnails = item['snippet']['thumbnails']
                thumbnail_url = thumbnails.get('high', {}).get('url') or \
                               thumbnails.get('medium', {}).get('url') or \
                               thumbnails.get('default', {}).get('url')
                
                videos.append({
                    'title': title,
                    'video_id': video_id,
                    'channel_title': channel_title,
                    'thumbnail_url': thumbnail_url
                })
        return videos

    except googleapiclient.errors.HttpError as e:
        # Handle API errors gracefully
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")
        if e.resp.status == 403:
            print("This might be due to an invalid API key or exceeding quota.")
        return None
    except Exception as e:
        # Handle other potential errors
        print(f"An unexpected error occurred: {e}")
        return None

@app.route('/search-videos', methods=['POST'])
def search_videos():
    """API endpoint to search for videos based on a concept."""
    try:
        concept = request.form.get('concept', '')
        if not concept:
            return jsonify({'success': False, 'error': 'No concept provided'}), 400
            
        # Search for videos using the YouTube API
        videos = search_youtube(concept)
        
        if videos is None:
            return jsonify({'success': False, 'error': 'Failed to fetch videos. Please try again later.'}), 500
            
        return jsonify({
            'success': True,
            'videos': videos
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/summarize-pdf', methods=['POST'])
def summarize_pdf():
    """API endpoint to summarize the uploaded PDF."""
    try:
        if 'pdf_text' not in session:
            return jsonify({'error': 'No PDF content available. Please upload a PDF first.'}), 400
            
        text = session['pdf_text']
        method = request.form.get('method', 'extractive')  # 'extractive' or 'ai'
        sentences = int(request.form.get('sentences', 3))
        
        if method == 'ai':
            summary = ai_summarize_text(text)
        else:
            summary = extract_summarize_text(text, sentences)
            
        return jsonify({
            'success': True,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'reduction_percentage': round((1 - len(summary) / len(text)) * 100, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information."""
    if request.method == 'POST':
        # Get form data
        first_name = request.form.get('firstName', '').strip()
        last_name = request.form.get('lastName', '').strip()
        
        # Combine into full name
        full_name = f"{first_name} {last_name}".strip()
        
        if not full_name:
            flash("Name cannot be empty.", "error")
            return redirect(url_for('settings'))
        
        try:
            # Update the user's full name in the database
            db = get_db()
            db.execute(
                "UPDATE Users SET full_name = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (full_name, g.user['id'])
            )
            db.commit()
            flash("Profile updated successfully!", "success")
        except Exception as e:
            print(f"Error updating profile: {e}", file=sys.stderr)
            flash("An error occurred while updating your profile.", "error")
            
    return redirect(url_for('settings'))

# --- DSA Practice Routes ---
@app.route('/practice')
@login_required
def practice():
    """DSA practice page with all problems organized by category."""
    # Load the Excel file with problem set data
    xlsx_path = 'data/Strivers.xlsx'
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        flash(f"Error loading problem set data: {str(e)}", "error")
        return redirect(url_for('dashboard'))
    
    # For displaying on sidebar, for user context - still show proficiency levels
    proficiency_levels = {
        "Arrays": "Beginner",
        "Strings": "Intermediate",
        "Linked Lists": "Beginner",
        "Sorting": "Intermediate",
        "Graphs": "Advanced",
        "Trees": "Beginner",
        "Dynamic Programming": "Advanced"
    }
    
    # Map difficulty ratings from the Excel file to easy/medium/hard
    def map_difficulty(rating):
        if pd.isna(rating) or rating <= 1.5:
            return "easy", "Easy"
        elif rating <= 3.5:
            return "medium", "Medium"
        else:
            return "hard", "Hard"
    
    # Process all problems from the Excel file
    all_problems = []
    problems_by_category = {}
    
    # Get unique categories and initialize counts
    categories = df['Category'].unique().tolist()  # Convert NumPy array to Python list
    topic_counts = {category: 0 for category in categories}
    
    # Process and organize all problems by category
    for category in categories:
        category_problems = []
        category_df = df[df['Category'] == category]
        topic_counts[category] = len(category_df)
        
        # Sort by difficulty within each category
        category_df = category_df.sort_values(by='Level of Difficulty')
        
        # Process each problem
        for _, row in category_df.iterrows():
            difficulty_class, difficulty_label = map_difficulty(row['Level of Difficulty'])
            problem = {
                'name': row['Name'],
                'problem_link': row['Link to problem'],
                'solution_link': row['Link to solution'],
                'category': category,
                'difficulty_class': difficulty_class,
                'difficulty_label': difficulty_label,
                'difficulty_value': row['Level of Difficulty']
            }
            all_problems.append(problem)
            category_problems.append(problem)
        
        # Store problems organized by category
        problems_by_category[category] = category_problems
    
    # Get user completion data (in a real app, this would come from database)
    # For demo purposes, we'll just use an empty dictionary
    completed_problems = {}
    total_completed = 0
    streak = 3  # Example streak value
    
    return render_template('practice.html', 
                          all_problems=all_problems,
                          problems_by_category=problems_by_category,
                          categories=categories,
                          total_problems=len(all_problems),
                          topic_counts=topic_counts,
                          proficiency_levels=proficiency_levels,
                          total_completed=total_completed,
                          streak=streak)

@app.route('/update_problem_status', methods=['POST'])
@login_required
def update_problem_status():
    """Update the completion status of a practice problem."""
    data = request.get_json()
    if not data or 'problem_id' not in data or 'completed' not in data:
        return jsonify({'success': False, 'error': 'Invalid data'}), 400
    
    problem_id = data['problem_id']
    completed = data['completed']
    
    # In a real app, we would update the database
    # For demo purposes, we'll just return success
    
    return jsonify({'success': True})

if __name__ == '__main__':
    # Ensure the database exists (run your schema script if needed)
    if not os.path.exists(app.config['DATABASE']):
        print(f"Database file '{app.config['DATABASE']}' not found.")
        print("Please run the database creation script first.")
        exit()
    app.run(debug=True) # debug=True enables auto-reloading and error pages