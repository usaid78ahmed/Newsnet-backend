import json
from pymongo import MongoClient

# Connect to MongoDB (adjust the connection string if needed)
client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
collection = db["entity_relationships"]

# Load the JSON file
with open("entity_relationships.json", "r") as f:
    data = json.load(f)

# Prepare and clean documents for insertion.
documents = []
for key, value in data.items():
    # Convert key to integer if possible.
    try:
        doc_id = int(key)
    except ValueError:
        doc_id = key

    # Retrieve the raw relationship triplets
    raw_triplets = value.get("relationships", [])
    cleaned_triplets = []
    for triplet in raw_triplets:
        # Remove backslashes from each element and perform any additional cleaning
        cleaned = [elem.replace("\\", "").strip() for elem in triplet]
        cleaned_triplets.append(cleaned)

    # Create the document with cleaned relationships.
    document = {
        "id": doc_id,
        "relationships": cleaned_triplets
    }
    documents.append(document)

# Insert the documents into MongoDB.
if documents:
    result = collection.insert_many(documents)
    print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
else:
    print("No documents to insert.")
