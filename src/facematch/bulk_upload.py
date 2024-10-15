import os 
import csv
import json 

def upload_embedding_to_database(data):
    csv_file = "data/embeddings.csv"
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    with open(csv_file, mode='w', newline='') as file:
        # Create a CSV writer object
        writer = csv.DictWriter(file, fieldnames=data[0].keys())

        # Write the header (column names)
        writer.writeheader()

        # Write each dictionary as a row in the CSV file
        writer.writerows(data)