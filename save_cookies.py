import asyncio
from twikit import Client

from dotenv import load_dotenv
load_dotenv()
import os

USERNAME = os.getenv('USERNAME')
EMAIL = os.getenv('EMAIL')
PASSWORD = os.getenv('PASSWORD')

# Initialize client
client = Client('en-US')

async def main():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file='cookies.json'
    )

asyncio.run(main())