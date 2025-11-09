import google.generativeai as genai

API_KEY = "AIzaSyAgN9j9mGNVE7sbDHrwt1vQ6Q_8QoHgYDM"  # ⬅️ Paste your key here manually

genai.configure(api_key=API_KEY)

print("Listing available models...\n")

try:
    for model in genai.list_models():
        print(model.name)
except Exception as e:
    print("Error listing models:", e)
