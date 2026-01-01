from google import genai
import os

print("API KEY FOUND:", bool(os.environ.get("GOOGLE_API_KEY")))

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain why engineering is a stable career for parents."
)

print("\n--- GEMINI RESPONSE ---\n")
print(response.text)
