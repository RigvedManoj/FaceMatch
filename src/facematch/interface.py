import json
import os

from src.facematch.database_functions import upload_embedding_to_database
from src.facematch.face_representation import detect_faces_and_get_embeddings
from src.facematch.resource_path import get_resource_path
from src.facematch.similarity_search import cosine_similarity_search


class FaceMatchModel:
    # Function that takes in path to directory of images to upload to database and returns a success or failure message.
    def bulk_upload(self, image_directory_path, database_path=None):
        try:
            if database_path is None:
                config_path = get_resource_path("db_config.json")
                with open(config_path, "r") as config_file:
                    config = json.load(config_file)
                database_path = get_resource_path(config["database_path"])

            embedding_outputs = []
            for filename in os.listdir(image_directory_path):
                image_path = os.path.join(image_directory_path, filename)
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                    embedding_outputs.extend(
                        detect_faces_and_get_embeddings(image_path)
                    )

            upload_embedding_to_database(embedding_outputs, database_path)

            return "Successfully uploaded to database"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    # Function that takes in path to image and returns all images that have the same person.
    def find_face(self, image_file_path, database_path=None):
        try:
            if database_path is None:
                config_path = get_resource_path("db_config.json")
                with open(config_path, "r") as config_file:
                    config = json.load(config_file)
                database_path = get_resource_path(config["database_path"])
            filename = os.path.basename(image_file_path)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                embedding_outputs = detect_faces_and_get_embeddings(image_file_path)
                matching_image_paths = []
                for embedding_output in embedding_outputs:
                    output = cosine_similarity_search(
                        embedding_output["embedding"], database_path, threshold=0.6
                    )
                    matching_image_paths.extend(output)
                return matching_image_paths
            else:
                return "Error: Provided file is not of image type"
        except Exception as e:
            return f"An error occurred: {str(e)}"
