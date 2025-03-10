import asyncio
from twikit import Client
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
USERNAME = os.environ.get('TWITTER_USERNAME')
EMAIL = os.environ.get('TWITTER_EMAIL')
PASSWORD = os.environ.get('TWITTER_PASSWORD')

# Initialize client
client = Client('en-US')

async def main():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file='cookies.json'
    )

asyncio.run(main())``