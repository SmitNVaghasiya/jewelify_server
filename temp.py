# from pymongo import MongoClient
# import os
# from dotenv import load_dotenv

# load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")
# client = MongoClient(MONGO_URI)
# db = client["jewelify"]
# collection = db["recommendations"]
# prediction = collection.find().sort("timestamp", -1).limit(1)[0]
# print("Latest prediction:", prediction)
# client.close()

import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
print("DEBUG MONGO_URI:", MONGO_URI)

client = MongoClient(MONGO_URI)
db = client["jewelify"]  # Or your actual DB name
collection = db["recommendations"]  # Or the actual collection name

# Check how many documents are there
count = collection.count_documents({})
print(f"Number of documents in 'recommendations': {count}")

for doc in collection.find():
    print(doc)
