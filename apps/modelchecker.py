import os
from dotenv import load_dotenv
from google import genai

load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Embedding-capable models:")

for m in client.models.list():
    actions = set(m.supported_actions or [])

    if "embedContent" in actions:
        print(f"- {m.name}")
        print(f"  display name: {m.display_name}")
        print(f"  actions: {sorted(actions)}")