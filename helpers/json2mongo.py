from pymongo import MongoClient
import json
from tqdm import tqdm
from config import MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME

def json2mongo(json_file_path, mongo_uri, mongo_db_name, mongo_collection_name):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    client = MongoClient(mongo_uri)
    db = client[mongo_db_name]
    collection = db[mongo_collection_name]

    for i in tqdm(range(0, len(data), 1000), desc="Inserting data into MongoDB"):
        collection.insert_many(data[i:i+1000])
    
    print(f"Inserted {len(data)} documents into MongoDB")

if __name__ == "__main__":
    json2mongo(json_file_path="../data/processed_data.jsonl", mongo_uri=MONGO_URI, mongo_db_name=MONGO_DB_NAME, mongo_collection_name=MONGO_COLLECTION_NAME)