import json

from deepface import DeepFace

from src.facematch.resource_path import get_resource_path


# Function that takes in path to image and returns a list of face embeddings and corresponding region for all
# faces in the image.
def detect_faces_and_get_embeddings(image_path):
    config_path = get_resource_path("model_config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    model_name = config["model_name"]
    detector_backend = config["detector_backend"]

    results = DeepFace.represent(
        image_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
    )
    face_embeddings = []

    for i, result in enumerate(results):
        embedding = result["embedding"]

        x, y, width, height = (
            result["facial_area"]["x"],
            result["facial_area"]["y"],
            result["facial_area"]["w"],
            result["facial_area"]["h"],
        )

        face_embeddings.append(
            {
                "image_path": image_path,
                "embedding": embedding,
                "bbox": [x, y, width, height],
            }
        )

    return face_embeddings
