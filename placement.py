import requests

# Function to fetch internships from Adzuna API for India
def fetch_internships_from_adzuna(skills, location="Karnataka", num_results=5):
    # URL for the Adzuna API (India-based search)
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    
    # Your Adzuna App ID and API Key (replace with your credentials)
    params = {
        "app_id": "96e12eac",  # Replace with your App ID
        "app_key": "7a545dc457029cd2527a9f21a366010e",  # Replace with your API Key
        "what": skills,  # Search for internships related to skills
        "where": location,  # Search by location (e.g., Karnataka, India)
        "results_per_page": num_results,  # Number of results per page
        "content-type": "application/json",
    }

    # Send the request to the Adzuna API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        internships = response.json().get("results", [])
        if internships:
            # Display internship details
            print(f"Found {len(internships)} internships matching your skills in {location}:")
            for idx, internship in enumerate(internships, 1):
                print(f"\n{idx}. Job Title: {internship['title']}")
                print(f"   Company: {internship['company']['display_name']}")
                print(f"   Location: {', '.join(internship['location']['area'])}")
                print(f"   Job Link: {internship['redirect_url']}")
                print("-" * 50)
        else:
            print(f"No internships found in {location} matching the skills.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Main function to take user input and fetch internships
def main():
    # Ask user for the skills they have
    skills = input("Enter your skills (e.g., Python, Data Science): ")
    
    # Ask user for the location preference (default to Karnataka)
    location = input("Enter location (default is 'Karnataka'): ")
    if not location:
        location = "Karnataka"
    
    # Ask user for the number of results (default to 5)
    num_results = input("Enter number of results you want (default is 5): ")
    if not num_results:
        num_results = 5
    else:
        num_results = int(num_results)
    
    # Call the function to fetch internships
    fetch_internships_from_adzuna(skills, location, num_results)

# Run the main function
if __name__== "__main__":
    main()