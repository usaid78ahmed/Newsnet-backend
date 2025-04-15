from pymongo import MongoClient

# Connect to MongoDB running on localhost.
client = MongoClient("mongodb://localhost:27017/")

# Create (or get) a database named 'mydatabase'
db = client["mydatabase"]

# Create (or get) a collection named 'articles'
collection = db["articles"]

# Insert a sample document into the 'articles' collection.
sample_document = {
    "title": "Hello, MongoDB!",
    "content": "This is a test document.",
    "tags": ["example", "mongodb", "pymongo"]
}

result = collection.insert_one(sample_document)
print("Inserted document with id:", result.inserted_id)

# Optionally, list all databases to verify creation.
print("Databases:", client.list_database_names())
