# from pymongo import MongoClient
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Get MongoDB URI from environment variables
# MONGO_URI = os.getenv("MONGO_URI")
# if not MONGO_URI:
#     raise ValueError("MONGO_URI not found in environment variables")

# # Connect to MongoDB
# client = MongoClient(MONGO_URI)
# db = client["jewelify"]
# collection = db["recommendations"]

# # Fetch and print all documents in the recommendations collection
# print("Contents of the 'recommendations' collection:")
# documents = collection.find()
# for doc in documents:
#     print(doc)

# # Close the connection
# client.close()