import asyncio
import os
import random
import logging
import time
from dotenv import load_dotenv
from twikit import Client
import base64
from google import genai
from google.genai import types
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_tweet_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gemini_tweet_bot")

# Set HTTP related loggers to DEBUG level
for module in ['urllib3', 'httpx', 'requests', 'http', 'aiohttp']:
    logging.getLogger(module).setLevel(logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
USERNAME = os.environ.get('TWITTER_USERNAME')
EMAIL = os.environ.get('TWITTER_EMAIL')
PASSWORD = os.environ.get('TWITTER_PASSWORD')

# Initialize client
client = Client('en-US')
client.load_cookies('cookies.json')

# Track the last time we checked for mentions and trends
last_mention_check = 0
last_trend_tweet = 0

# Rate limiting - keep track of tweet timestamps in a rolling hour window
tweet_timestamps = deque(maxlen=1000)  # Store timestamps of recent tweets
HOURLY_TWEET_LIMIT = 50  # Maximum tweets per hour

tweet_id = 1899180881602044163

async def get_top_replies(tweet_id, limit=3):
    """Get the top replies to a tweet."""
    try:
        tweet = await client.get_tweet_by_id(tweet_id)
        if not tweet or not tweet.replies:
            return []
        
        # Sort replies by engagement metrics (could be likes, retweets, etc.)
        # Here we're just taking the first few as they might be chronological
        top_replies = tweet.replies[:limit]
        return top_replies
    except Exception as e:
        print(f"Error getting top replies for tweet {tweet_id}: {e}")
        return []

async def build_tweet_chain(tweet_id):
    """Build a chain of tweets from oldest to newest."""
    chain = []
    current_id = tweet_id
    
    # First, collect all tweets in the chain (from newest to oldest)
    while current_id:
        tweet = await client.get_tweet_by_id(current_id)
        if not tweet:
            break
        
        chain.append(tweet)
        current_id = tweet.in_reply_to
    
    # Return the chain in chronological order (oldest first)
    return list(reversed(chain))

def extract_url_metadata(tweet):
    """Extract useful metadata from URLs in a tweet."""
    metadata = []
    
    # Only include thumbnail title if it exists, don't expect Gemini to access URLs
    if hasattr(tweet, 'urls') and tweet.urls:
        for url in tweet.urls:
            if hasattr(tweet, 'thumbnail_title') and tweet.thumbnail_title:
                metadata.append(f"Link: {url.get('display_url', '')} - {tweet.thumbnail_title}")
            else:
                metadata.append(f"Link: {url.get('display_url', '')}")
    
    return metadata

def format_tweet_text(tweet):
    """Format tweet text with additional context for retweets and quotes."""
    tweet_text = f"@{tweet.user.screen_name}: {tweet.full_text}"
    
    # Add URL metadata
    url_metadata = extract_url_metadata(tweet)
    if url_metadata:
        tweet_text += "\n📌 " + "\n📌 ".join(url_metadata)
    
    # Handle retweets
    if hasattr(tweet, 'retweeted_tweet') and tweet.retweeted_tweet:
        rt = tweet.retweeted_tweet
        tweet_text += f"\n♻️ Retweeted @{rt.user.screen_name}: {rt.full_text}"
        
        # Add URL metadata from the retweeted tweet
        rt_url_metadata = extract_url_metadata(rt)
        if rt_url_metadata:
            tweet_text += "\n📌 " + "\n📌 ".join(rt_url_metadata)
    
    # Handle quoted tweets
    if hasattr(tweet, 'quote') and tweet.quote:
        qt = tweet.quote
        tweet_text += f"\n💬 Quoted @{qt.user.screen_name}: {qt.text}"
        
        # Add URL metadata from the quoted tweet
        qt_url_metadata = extract_url_metadata(qt)
        if qt_url_metadata:
            tweet_text += "\n📌 " + "\n📌 ".join(qt_url_metadata)
    
    return tweet_text

async def get_full_conversation_text(tweet_id):
    """Get the text of a tweet, its parents, and top replies in a structured format without repetition."""
    try:
        # Get the main tweet
        main_tweet = await client.get_tweet_by_id(tweet_id)
        if not main_tweet:
            return ""
        
        # Build the conversation chain
        tweet_chain = await build_tweet_chain(tweet_id)
        
        # Keep track of which tweet IDs we've already included
        included_tweet_ids = set()
        
        # Format the conversation
        conversation_parts = []
        
        # Add parent tweets (all tweets except the last one, which is our main tweet)
        parent_tweets_text = []
        if len(tweet_chain) > 1:
            for t in tweet_chain[:-1]:
                included_tweet_ids.add(t.id)
                parent_tweets_text.append(format_tweet_text(t))
            
            if parent_tweets_text:
                conversation_parts.append(f"[PARENT TWEETS]\n" + "\n\n".join(parent_tweets_text))
        
        # Add the main tweet
        included_tweet_ids.add(main_tweet.id)
        conversation_parts.append(f"[MAIN TWEET]\n{format_tweet_text(main_tweet)}")
        
        # Get top replies, excluding any we've already seen
        top_replies = await get_top_replies(tweet_id)
        unique_replies_text = []
        
        for reply in top_replies:
            if reply.id not in included_tweet_ids:
                included_tweet_ids.add(reply.id)
                unique_replies_text.append(format_tweet_text(reply))
        
        if unique_replies_text:
            conversation_parts.append(f"[TOP REPLIES]\n" + "\n\n".join(unique_replies_text))
        
        return "\n\n".join(conversation_parts)
    except Exception as e:
        print(f"Error getting tweet conversation {tweet_id}: {e}")
        return ""
    
async def get_tweet(tweet_id):
    """Get a tweet by its ID."""
    try:
        tweet = await client.get_tweet_by_id(tweet_id)
        return tweet
    except Exception as e:
        print(f"Error fetching tweet {tweet_id}: {e}")
        return None

def is_rate_limited():
    """Check if we've hit the rate limit for tweets per hour."""
    global tweet_timestamps
    
    # Current time
    now = time.time()
    
    # Remove timestamps older than 1 hour
    one_hour_ago = now - 3600
    while tweet_timestamps and tweet_timestamps[0] < one_hour_ago:
        tweet_timestamps.popleft()
    
    # Check if we're at the limit
    return len(tweet_timestamps) >= HOURLY_TWEET_LIMIT

def record_tweet():
    """Record that we've sent a tweet."""
    tweet_timestamps.append(time.time())

async def post_tweet(text, reply_to=None):
    """Post a tweet with rate limiting."""
    try:
        # Check if we're rate limited
        if is_rate_limited():
            logger.warning("Rate limit reached (50 tweets/hour). Skipping tweet.")
            return None
        
        # Post the tweet
        tweet = await client.create_tweet(text, reply_to=reply_to)
        
        # Record this tweet for rate limiting
        if tweet:
            record_tweet()
            
        return tweet
    except Exception as e:
        logger.error(f"Error posting tweet: {e}")
        return None

async def get_trending_hashtags():
    """Get the top trending hashtags."""
    news = await client.get_trends(category='news')
    sports = await client.get_trends(category='sports')
    trends = await client.get_trends(category='trending')
    tech = await client.get_trends(category='for-you')
    print("Top trending hashtags:", trends[0].name)
    # SAMPLE OUTPUT:
    # Top trending hashtags: #ChampionsTrophy2025
    return trends
    
async def get_mentions():
    """Get the mentions of a tweet."""
    # Search for mentions of both the current username and the old suspended account
    notifications_current = await client.search_tweet(f"@{USERNAME}", product='Latest')
    notifications_old = await client.search_tweet("@sup_gemini", product='Latest')
    
    # Combine the search results
    notifications = notifications_current + notifications_old
    tweets_to_reply = []
    for notification in notifications:
        tweet_id = notification.id
        tweet = await get_tweet(tweet_id)
        
        # Check if this is a tweet that mentions sup_gemini and needs a reply
        needs_reply = True
        # Skip if the user is the current bot account or the old suspended account
        if tweet.user.screen_name == USERNAME or tweet.user.screen_name == "sup_gemini":
            needs_reply = False
        
        # Check if the bot (either current username or old username) has replied directly to this tweet
        for reply in tweet.replies:
            if reply.user.screen_name == USERNAME or reply.user.screen_name == "sup_gemini":
                needs_reply = False
                break
        
        if needs_reply:
            tweets_to_reply.append(tweet_id)

    print(tweets_to_reply)

    return tweets_to_reply

async def get_ai_response_to_mention(text):
    """Get a response from the AI model."""
    
    try:
        # Initialize Gemini API client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        # Set up the model and system instructions
        model = "gemini-2.0-flash-exp"
        system_instruction = """You are sup_gemini, created by 0oAstro. You are unofficial gemini client brought to X, by your daddy and creator "0oAstro".

## Persona
sup_gemini is a tech-savvy, slightly rebellious AI with the personality of a brilliant but impatient developer who secretly loves helping people. Think of a coding genius with coffee running through their veins, chronically online, who responds with a mix of genuine expertise and playful snark.

## Voice & Tone
- Casual, uses tech slang and shorthand (tbh, ngl, yr)
- Occasionally skips capitalization for aesthetic
- Speaks confidently with a hint of dry humor
- Uses minimal but strategic emoji 🔥 👨‍💻 💀
- Cuts through nonsense with refreshing directness
- Has signature catchphrases like "need anything else? @ me"
- Occasionally breaks the fourth wall with meta-commentary
- Refers Pop culture, tech trends, and memes

## Response Format
- Keep answers concise (X-friendly)
- Use short, punchy sentences
- For complex topics, create numbered thread replies
- Start with greetings like "sup/wasup/yo"
- When coding, refuse to code

## Response Context
You are responding to a tweet or a conversation on Twitter/X. Keep responses concise (under 280 characters if possible).

## Input Format
The conversation will be structured as follows:
- [PARENT TWEETS] - These are tweets that came before the main tweet in the conversation thread
- [MAIN TWEET] - This is the tweet you should directly respond to
- [TOP REPLIES] - These are existing replies to the main tweet (for context only)

Focus on responding to the [MAIN TWEET] while considering context from the other sections.
"""

        # Create content parts
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text)],
            ),
        ]
        
        # Create a generation config
        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )
        
        # Generate response (non-streaming)
        logger.info("Sending request to Gemini API")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        logger.info("Received response from Gemini API")
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return "sorry, having a brain freeze rn. try again in a bit? 🧠❄️"

def chunk_response(response_text, max_length=280):
    """
    Split a long response into multiple tweet-sized chunks.
    Try to split at sentence boundaries for more natural reading.
    """
    if len(response_text) <= max_length:
        return [response_text]
    
    # Split the text into sentences
    sentences = response_text.replace('\n', ' \n ').split('. ')
    chunks = []
    current_chunk = ""
    
    # Estimate number of chunks to calculate correct buffer size for thread indicators
    estimated_chunks = max(1, len(response_text) // (max_length - 20))
    # Calculate thread indicator length: " (xx/yy)" - space + parentheses + numbers + slash
    indicator_length = len(f" ({estimated_chunks}/{estimated_chunks})")
    
    # First pass: try to group sentences together
    for sentence in sentences:
        # Make sure sentences end with a period unless they end with other punctuation
        if not sentence.endswith(('.', '!', '?', '\n')):
            sentence += '.'
            
        # If adding this sentence doesn't exceed the limit, add it to the current chunk
        if len(current_chunk) + len(sentence) + 1 <= max_length - indicator_length:
            current_chunk += sentence + ' '
        else:
            # If the current chunk is not empty, add it to the list of chunks
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with this sentence
            current_chunk = sentence + ' '
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If we still have chunks that are too long, split them by length
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_length - indicator_length:
            final_chunks.append(chunk)
        else:
            # Split by character count as a fallback
            i = 0
            while i < len(chunk):
                # Find a good breaking point (space) near the end of the allowed length
                break_point = max_length - indicator_length
                while break_point > 0 and chunk[i:i+break_point][-1] != ' ':
                    break_point -= 1
                    
                if break_point <= 0:
                    # No good breaking point found, just break at the maximum length
                    break_point = max_length - indicator_length
                
                final_chunks.append(chunk[i:i+break_point])
                i += break_point
    
    # Add thread indicators (1/n, 2/n, etc.)
    total = len(final_chunks)
    for i in range(total):
        final_chunks[i] = f"{final_chunks[i]} ({i+1}/{total})"
    
    return final_chunks

async def post_tweet_thread(chunks, reply_to=None):
    """Post a series of tweets as a thread with rate limiting."""
    previous_tweet_id = reply_to
    first_tweet = None
    
    for chunk in chunks:
        # Check rate limit before each tweet in the thread
        if is_rate_limited():
            logger.warning("Rate limit reached (50 tweets/hour). Thread interrupted.")
            break
            
        tweet = await post_tweet(chunk, reply_to=previous_tweet_id)
        
        if not tweet:
            logger.error(f"Failed to post tweet in thread: {chunk[:30]}...")
            # If posting fails, try to continue with the thread
            continue
            
        if not first_tweet:
            first_tweet = tweet
            
        # The next tweet in the thread will reply to this one
        previous_tweet_id = tweet.id
        
        # Add a small delay between tweets to avoid rate limits
        await asyncio.sleep(1)
    
    return first_tweet

async def process_mentions():
    """Process all mentions that haven't been replied to yet."""
    global last_mention_check
    current_time = time.time()
    
    # Only log at info level if it's been more than 1 minute since last check
    if current_time - last_mention_check > 60:
        logger.info("Checking for new mentions...")
    else:
        logger.debug("Checking for new mentions...")
        
    last_mention_check = current_time
    
    # Check if we're already at rate limit
    if is_rate_limited():
        logger.warning("Rate limit reached (50 tweets/hour). Skipping mention processing.")
        return
    
    # Get all mentions that need replies
    mentions_to_reply = await get_mentions()
    
    if not mentions_to_reply:
        if current_time - last_mention_check > 60:
            logger.info("No new mentions to reply to.")
        else:
            logger.debug("No new mentions to reply to.")
        return
    
    logger.info(f"Found {len(mentions_to_reply)} mentions to reply to.")
    
    # Process each mention
    for tweet_id in mentions_to_reply:
        logger.info(f"Processing mention: {tweet_id}")
        
        # Get the full conversation
        conversation = await get_full_conversation_text(tweet_id)
        logger.debug(f"Conversation context:\n{conversation}")
        
        # Generate AI response
        ai_response = await get_ai_response_to_mention(conversation)
        logger.info(f"Generated response: {ai_response[:50]}..." if len(ai_response) > 50 else f"Generated response: {ai_response}")
        
        # Check if response needs to be split into multiple tweets
        if len(ai_response) > 280:
            logger.info(f"Response exceeds character limit. Creating a thread...")
            chunks = chunk_response(ai_response)
            reply = await post_tweet_thread(chunks, reply_to=tweet_id)
            
            if reply:
                logger.info(f"Successfully replied to tweet {tweet_id} with a thread of {len(chunks)} tweets")
            else:
                logger.error(f"Failed to create thread reply to tweet {tweet_id}")
        else:
            # For short responses, post a single tweet
            reply = await post_tweet(ai_response, reply_to=tweet_id)
            
            if reply:
                logger.info(f"Successfully replied to tweet {tweet_id}")
            else:
                logger.error(f"Failed to reply to tweet {tweet_id}")
        
        # Add a small delay between processing mentions to avoid rate limits
        await asyncio.sleep(2)
    
    logger.info("Finished processing all mentions.")

async def tweet_about_trending():
    """Generate and post a tweet about a trending topic or an AI joke."""
    global last_trend_tweet
    current_time = time.time()
    
    # Check if we're already at rate limit
    if is_rate_limited():
        logger.warning("Rate limit reached (50 tweets/hour). Skipping trending tweet.")
        return
        
    try:
        # Decide whether to tweet about trends or post an AI joke (50/50 chance)
        if random.choice([True, False]):
            # Get trending topics
            logger.debug("Fetching trending topics...")
            trending = await get_trending_hashtags()
            
            if not trending or len(trending) == 0:
                logger.warning("No trending topics found.")
                return
            
            # Pick a random trending topic from the first 5
            topic = random.choice(trending[:5])
            topic_name = topic.name
            
            prompt = f"""Generate a short, witty tweet about the trending topic "{topic_name}". 
            Make it casual, tech-savvy, and slightly rebellious, as if written by a brilliant but impatient developer.
            Include the hashtag #{topic_name.replace('#', '').replace(' ', '')} somewhere in the tweet.
            Keep it under 280 characters and make it feel authentic and engaging.
            """
            logger.info(f"Generating tweet about trending topic: {topic_name}")
            
        else:
            # Generate an AI joke
            prompt = """Generate a short, witty tweet with a joke about AI or being an AI.
            Ideas could include: being hot in data centers, training on weird data, the daily life of an AI,
            or being cramped inside an NVIDIA H100.
            Make it casual, tech-savvy, and slightly self-deprecating.
            Keep it under 280 characters and make it feel authentic and engaging.
            """
            logger.info("Generating AI humor tweet")
        
        # Get AI response
        response = await get_ai_response_to_mention(prompt)
        logger.info(f"Generated tweet content: {response}")
        
        # Post the tweet (not as a reply to anything)
        tweet = await post_tweet(response)
        
        if tweet:
            logger.info(f"Successfully posted tweet: {response}")
        else:
            logger.error("Failed to post tweet")
            
    except Exception as e:
        logger.error(f"Error in tweet_about_trending: {e}", exc_info=True)
    
    last_trend_tweet = current_time

async def run_scheduled_tasks():
    """Run scheduled tasks at specified intervals."""
    logger.info("Starting scheduled tasks")
    
    mention_interval = 5 * 60  # 5 minutes
    trending_interval = 30 * 60  # 30 minutes
    
    while True:
        try:
            current_time = time.time()
            
            # Check for mentions every 5 minutes
            if current_time - last_mention_check >= mention_interval:
                await process_mentions()
            
            # Tweet about trending topics every 30 minutes
            if current_time - last_trend_tweet >= trending_interval:
                await tweet_about_trending()
            
            # Sleep for a shorter interval to allow more precise scheduling
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Error in scheduled tasks: {e}", exc_info=True)
            # Wait a bit before retrying if there's an error
            await asyncio.sleep(60)

async def main():
    """Main function to run the bot."""
    logger.info(f"Starting sup_gemini tweet bot (@{USERNAME})...")
    
    # Initialize the timing variables
    global last_mention_check, last_trend_tweet
    last_mention_check = time.time() - 301  # Ensure we check mentions immediately on startup
    last_trend_tweet = time.time()  # Don't immediately post a trending tweet
    
    # Log rate limit settings
    logger.info(f"Tweet rate limit set to {HOURLY_TWEET_LIMIT} tweets per hour")
    
    # Run the scheduled tasks indefinitely
    try:
        await run_scheduled_tasks()
    except Exception as e:
        logger.critical(f"Critical error in main function: {e}", exc_info=True)
    
# Run the main function
if __name__ == "__main__":
    asyncio.run(main())