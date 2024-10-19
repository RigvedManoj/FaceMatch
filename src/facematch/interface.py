import json
import os

import face_representation as fr
import database_functions as dbf
import similarity_search as ss


# Function that takes in path to directory of images to upload to database and returns a success or failure message.
def bulk_upload(image_directory_path, database_path=None):
    try:
        if database_path is None:
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, "../..", "resources", "db_config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            database_path = config["database_path"]

        for filename in os.listdir(image_directory_path):
            image_path = os.path.join(image_directory_path, filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                embedding_outputs = fr.detect_faces_and_get_embeddings(image_path)
                dbf.upload_embedding_to_database(embedding_outputs, database_path)
        return "Successfully uploaded to database"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Function that takes in path to image and returns all images that have the same person.
def match_face(image_file_path, database_path=None):
    try:
        if database_path is None:
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, "../..", "resources", "db_config.json")
            with open(config_path, "r") as config_file:
                config = json.load(config_file)
            database_path = config["database_path"]
        filename = os.path.basename(image_file_path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            embedding_outputs = fr.detect_faces_and_get_embeddings(image_file_path)
            matching_image_paths = []
            for embedding_output in embedding_outputs:
                matching_image_paths.extend(ss.cosine_similarity_search(embedding_output, database_path, threshold=0.6))
            return matching_image_paths
        else:
            return "Error: Provided file is not of image type"
    except Exception as e:
        return f"An error occurred: {str(e)}"
