import os
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import load_dotenv

def search_youtube(api_key, query, max_results=6):
    """
    Searches YouTube for videos based on a query using YouTube Data API v3.

    Args:
        api_key (str): Your YouTube Data API v3 key.
        query (str): The search term or concept.
        max_results (int): The maximum number of results to return.

    Returns:
        list: A list of dictionaries, each containing video details
              (title, video_id, channel_title), or None if an error occurs.
    """
    try:
        api_service_name = "youtube"
        api_version = "v3"

        # Build the YouTube service object (variable name is 'youtube')
        # Adding static_discovery=False and additional parameters to bypass referrer restrictions
        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=api_key,
            static_discovery=False)

        # Make the API call using the 'youtube' variable (lowercase)
        request = youtube.search().list(
            part="snippet",         # Request basic details
            q=query,                # The search query from user
            type="video",           # Search only for videos
            maxResults=max_results  # Limit the number of results
        )
        response = request.execute()

        videos = []
        if 'items' in response:
            for item in response['items']:
                # Extract relevant information from the response snippet
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                channel_title = item['snippet']['channelTitle']
                
                # Extract thumbnail URLs (multiple sizes available)
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
        print(f"\nAn HTTP error {e.resp.status} occurred:\n{e.content}")
        if e.resp.status == 403:
             print("This might be due to an invalid API key or exceeding quota.")
        return None
    except Exception as e:
        # Handle other potential errors
        print(f"\nAn unexpected error occurred INSIDE search_youtube: {e}") # Added location info
        return None

def display_results(results):
    """Prints the formatted search results to the console."""
    if not results:
        print("No results found or an error occurred during the search.")
        return

    print("\n--- YouTube Video Results ---")
    for i, video in enumerate(results, 1):
        print(f"\n{i}. Title: {video['title']}")
        print(f"   Channel: {video['channel_title']}")
        # Construct the standard YouTube watch link
        print(f"   Link: https://www.youtube.com/watch?v={video['video_id']}") # Corrected Link
    print("---------------------------\n")
    
def generate_html(results, search_query):
    """Generates an HTML page with the search results and thumbnails.
    
    Args:
        results (list): List of video result dictionaries
        search_query (str): The search term used
        
    Returns:
        str: Path to the generated HTML file
    """
    if not results:
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YouTube Search Results</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                /* Base styling from the main theme */
                :root {
                    --bg-primary: #ffffff;
                    --bg-secondary: #f1f1f1;
                    --card-bg: #ffffff;
                    --text-primary: #000000;
                    --text-secondary: #444444;
                    --accent-red: #D81B27;
                    --accent-gold: #FFC72C;
                    --shadow-color: rgba(0, 0, 0, 0.1);
                    --border-color: #dcdcdc;
                }
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: 'Poppins', sans-serif;
                    line-height: 1.6;
                    background-color: var(--bg-secondary);
                    color: var(--text-primary);
                    padding: 20px;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1 {
                    font-size: 2rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                    color: var(--accent-red);
                }
                .error-message {
                    text-align: center;
                    padding: 2rem;
                    background-color: var(--card-bg);
                    border-radius: 8px;
                    box-shadow: 0 2px 10px var(--shadow-color);
                }
                .back-link {
                    display: block;
                    text-align: center;
                    margin-top: 2rem;
                    color: var(--accent-red);
                    text-decoration: none;
                }
                .back-link:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>YouTube Search Results</h1>
                <div class="error-message">
                    <p>No results found or an error occurred during the search.</p>
                </div>
                <a href="javascript:history.back()" class="back-link">Go Back</a>
            </div>
        </body>
        </html>
        """
    else:
        # Create HTML content with search results and thumbnails
        video_items = ""
        for video in results:
            video_items += f"""
            <div class="video-card">
                <div class="thumbnail">
                    <a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">
                        <img src="{video['thumbnail_url']}" alt="{video['title']}">
                    </a>
                </div>
                <div class="video-info">
                    <h3><a href="https://www.youtube.com/watch?v={video['video_id']}" target="_blank">{video['title']}</a></h3>
                    <p class="channel">{video['channel_title']}</p>
                </div>
            </div>
            """
            
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YouTube Search: {search_query}</title>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
            <style>
                /* Base styling from the main theme */
                :root {{
                    --bg-primary: #ffffff;
                    --bg-secondary: #f1f1f1;
                    --card-bg: #ffffff;
                    --text-primary: #000000;
                    --text-secondary: #444444;
                    --accent-red: #D81B27;
                    --accent-gold: #FFC72C;
                    --shadow-color: rgba(0, 0, 0, 0.1);
                    --border-color: #dcdcdc;
                }}
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Poppins', sans-serif;
                    line-height: 1.6;
                    background-color: var(--bg-secondary);
                    color: var(--text-primary);
                    padding: 20px;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1 {{
                    font-size: 2rem;
                    text-align: center;
                    margin-bottom: 1.5rem;
                    color: var(--accent-red);
                }}
                .search-info {{
                    text-align: center;
                    margin-bottom: 2rem;
                    color: var(--text-secondary);
                }}
                .video-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                    gap: 20px;
                }}
                .video-card {{
                    background-color: var(--card-bg);
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px var(--shadow-color);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .video-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 5px 15px var(--shadow-color);
                }}
                .thumbnail {{
                    position: relative;
                    width: 100%;
                    padding-top: 56.25%; /* 16:9 Aspect Ratio */
                    overflow: hidden;
                }}
                .thumbnail img {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }}
                .video-info {{
                    padding: 15px;
                }}
                .video-info h3 {{
                    font-size: 1rem;
                    margin-bottom: 8px;
                    line-height: 1.4;
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }}
                .video-info h3 a {{
                    color: var(--text-primary);
                    text-decoration: none;
                }}
                .video-info h3 a:hover {{
                    color: var(--accent-red);
                }}
                .video-info .channel {{
                    font-size: 0.9rem;
                    color: var(--text-secondary);
                }}
                .back-button {{
                    display: inline-block;
                    margin-bottom: 20px;
                    padding: 8px 16px;
                    background-color: var(--accent-red);
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    font-weight: 500;
                    transition: background-color 0.3s ease;
                }}
                .back-button:hover {{
                    background-color: #b01620;
                }}
                .search-form {{
                    display: flex;
                    margin-bottom: 2rem;
                    justify-content: center;
                }}
                .search-form input {{
                    padding: 10px 15px;
                    border: 1px solid var(--border-color);
                    border-radius: 4px 0 0 4px;
                    width: 60%;
                    max-width: 500px;
                    font-family: inherit;
                }}
                .search-form button {{
                    padding: 10px 15px;
                    background-color: var(--accent-red);
                    color: white;
                    border: none;
                    border-radius: 0 4px 4px 0;
                    cursor: pointer;
                    font-family: inherit;
                    font-weight: 500;
                }}
                .search-form button:hover {{
                    background-color: #b01620;
                }}
                @media (max-width: 768px) {{
                    .video-grid {{
                        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    }}
                    .search-form input {{
                        width: 70%;
                    }}
                }}
                @media (max-width: 480px) {{
                    .video-grid {{
                        grid-template-columns: 1fr;
                    }}
                    .search-form {{
                        flex-direction: column;
                        align-items: center;
                    }}
                    .search-form input {{
                        width: 100%;
                        border-radius: 4px;
                        margin-bottom: 10px;
                    }}
                    .search-form button {{
                        width: 100%;
                        border-radius: 4px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="javascript:history.back()" class="back-button"><i class="fas fa-arrow-left"></i> Back</a>
                <h1>YouTube Video Results</h1>
                <p class="search-info">Showing results for: <strong>{search_query}</strong></p>
                
                <form class="search-form" action="" method="post">
                    <input type="text" name="search_query" placeholder="Search for videos..." value="{search_query}">
                    <button type="submit"><i class="fas fa-search"></i> Search</button>
                </form>
                
                <div class="video-grid">
                    {video_items}
                </div>
            </div>
        </body>
        </html>
        """
    
    # Write to file
    html_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Templates', 'youtube_results.html')
    os.makedirs(os.path.dirname(html_file_path), exist_ok=True)
    
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return html_file_path

# --- Main execution block ---
if __name__ == "__main__":
    # --- Load environment variables from .env file ---
    load_dotenv()

    # --- Get API Key from environment ---
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YouTube API key not found.")
        print("Please ensure you have a .env file in the same directory")
        print("with the line: YOUTUBE_API_KEY='YOUR_API_KEY'")
        exit(1) # Exit if no key is found
    
    # Import webbrowser at the beginning to avoid multiple imports
    import webbrowser
    
    # Track if a browser window is already open
    browser_opened = False
    
    # Run continuously until user chooses to exit
    while True:
        # Clear terminal (Windows)
        os.system('cls')
        
        print("\n===== YouTube Video Search =====\n")
        print("Type 'exit' or 'quit' to close the application")
        
        # --- Get User Input ---
        search_query = input("\nEnter a concept to find videos for: ")
        
        # Check if user wants to exit
        if search_query.lower() in ['exit', 'quit']:
            print("\nThank you for using YouTube Video Search. Goodbye!")
            break
        
        if not search_query.strip():
            print("\nNo concept entered. Please try again.")
            input("Press Enter to continue...")
            continue
        
        # --- Perform Search ---
        print(f"\nSearching for '{search_query}' on YouTube...")
        search_results = search_youtube(api_key, search_query, max_results=6)
        
        # --- Display Results in Console ---
        display_results(search_results)
        
        # --- Generate HTML Page with Results ---
        if search_results:
            html_file_path = generate_html(search_results, search_query)
            print(f"\nHTML results page generated at: {html_file_path}")
            
            # Open the HTML file in the default browser
            webbrowser.open('file://' + os.path.abspath(html_file_path))
            browser_opened = True
        
        # Prompt before continuing to the next search
        input("\nPress Enter to search for another concept...")