import logging
from apify_client import ApifyClientAsync

# Initialize the logger
logging.basicConfig(level=logging.INFO)

# API Token for Apify
API_TOKEN = 'apify_api_iUkAGcxEEesRtRsfUTOrYQNYsX1Tex4kZKEZ'

async def fetch_captions(username, results_limit=5):
    """
    Fetch captions from an Instagram user using Apify's Instagram Post Scraper.

    Args:
        username (str): Instagram username to scrape.
        results_limit (int, optional): Number of posts to fetch. Defaults to 5.

    Returns:
        list[str]: List of post captions.
    """
    # Initialize Apify Client
    apify_client = ApifyClientAsync(API_TOKEN)

    # Start an Actor and wait for it to finish
    actor_client = apify_client.actor('apify/instagram-post-scraper')

    # Actor input
    actor_input = {
        "username": [username],
        "resultsLimit": results_limit
    }

    # Call the actor to fetch Instagram posts
    call_result = await actor_client.call(run_input=actor_input)

    if call_result is None:
        logging.error('Actor run failed.')
        return []

    # Fetch results from the Actor run's default dataset
    dataset_client = apify_client.dataset(call_result['defaultDatasetId'])
    list_items_result = await dataset_client.list_items()

    # Extract captions
    captions = []
    for item in list_items_result.items:
        caption = item.get("caption", "")
        if caption:
            captions.append(caption)

    logging.info(f"Fetched {len(captions)} captions from {username}.")
    return captions
