import os
import requests
from flask import Flask, render_template, request
from dotenv import load_dotenv
import random # For assigning default icons

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Adzuna API Credentials from environment variables
ADZUNA_APP_ID = os.environ.get("ADZUNA_APP_ID")
ADZUNA_API_KEY = os.environ.get("ADZUNA_API_KEY")

# Helper function to choose an icon based on title keywords
def get_job_icon(title):
    title_lower = title.lower()
    if "python" in title_lower:
        return "fab fa-python"
    elif "data science" in title_lower or "data scientist" in title_lower:
        return "fas fa-database"
    elif "analyst" in title_lower:
        return "fas fa-chart-line"
    elif "web" in title_lower or "frontend" in title_lower or "backend" in title_lower:
        return "fas fa-code"
    elif "engineer" in title_lower:
        return "fas fa-cogs"
    elif "manager" in title_lower:
        return "fas fa-user-tie"
    elif "security" in title_lower:
        return "fas fa-shield-alt"
    else:
        # Default generic icons
        return random.choice(["fas fa-briefcase", "fas fa-building", "fas fa-user-tag"])

def fetch_jobs_from_adzuna(skills, location="Karnataka", num_results=10):
    """Fetches job listings from Adzuna API."""
    if not ADZUNA_APP_ID or not ADZUNA_API_KEY:
        print("Error: Adzuna API credentials not found in environment variables.")
        return None, "API credentials missing."

    # Adzuna API endpoint for India
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "what": skills,
        "where": location,
        "results_per_page": num_results,
        "content-type": "application/json",
        # Optionally add more filters like "full_time": 1, "contract": 0 etc.
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()
        jobs = data.get("results", [])

        # Add icon to each job
        for job in jobs:
            job['icon'] = get_job_icon(job.get('title', ''))

        return jobs, None # Return jobs and no error

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Adzuna: {e}")
        return None, f"Network error or API issue: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, "An unexpected error occurred."

@app.route('/', methods=['GET', 'POST']) # Use root URL for simplicity
def placement_page():
    jobs = None
    error = None
    search_skills = ''
    search_location = ''

    if request.method == 'POST':
        search_skills = request.form.get('skills', '').strip()
        search_location = request.form.get('location', '').strip()
        location_query = search_location if search_location else "Karnataka" # Default if empty

        if not search_skills:
            error = "Please enter skills to search."
        else:
            jobs, error = fetch_jobs_from_adzuna(search_skills, location_query)

    # For GET requests or after POST, render the template
    # Pass search terms back to pre-fill the form
    return render_template('placement.html',
                           jobs=jobs,
                           error=error,
                           search_skills=search_skills,
                           search_location=search_location)

if __name__ == '__main__':
    app.run(debug=True) # debug=True for development, set to False for production