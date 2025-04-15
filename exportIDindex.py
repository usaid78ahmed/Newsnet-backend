import json
from pymongo import MongoClient

# Connect to MongoDB (adjust the connection string if needed)
client = MongoClient("mongodb://localhost:27017/") # Adjust IP if using WSL
db = client["mydatabase"]
collection = db["id_to_article"]

# Load the JSON file
with open("id_to_article_D.json", "r") as f:  # Replace with your actual file path
    articles_data = json.load(f)

# Convert JSON format to MongoDB format
documents = []
for article_id, article_info in articles_data.items():
    document = {
        "article_id": article_id,
        "title": article_info["title"],
        "content": article_info["content"],
        "link": article_info["link"],
        "date": article_info["date"]
    }
    documents.append(document)

# Insert into MongoDB
collection.insert_many(documents)

print(f"Inserted {len(documents)} articles into MongoDB.")
