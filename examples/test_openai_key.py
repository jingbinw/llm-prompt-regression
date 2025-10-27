"""
Quickly test if your OPENAI_API_KEY is working.
"""


import os
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv


async def test_openai_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return
    client = AsyncOpenAI(api_key=api_key)
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10,
        )
        print("Success! Response:", response.choices[0].message.content)
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    asyncio.run(test_openai_key())
