import argparse

from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

# Define the URL and set up client
IMAGE_STYLE_MATCH_MODEL_URL = "http://127.0.0.1:5000/findface"
client = MLClient(IMAGE_STYLE_MATCH_MODEL_URL)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="To parse text arguments")
parser.add_argument(
    "file_paths", metavar="file", type=str, nargs="+", help="Path to images"
)
args = parser.parse_args()

# Prepare inputs based on command line arguments
data_type = DataTypes.IMAGE
inputs = [{"file_path": file_path} for file_path in args.file_paths]

print(inputs)
# Send request to the model and print the response
response = client.request(inputs, data_type)
print("Response:")
print(response)
