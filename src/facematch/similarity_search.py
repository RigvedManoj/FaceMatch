import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_search(
    query_vector, database_filepath, top_n=None, threshold=None
):
    # Ensure at least one of top_n or threshold is set
    if top_n is None and threshold is None:
        raise ValueError("Either top_n or threshold must be specified.")

    csv_file = database_filepath

    # Read the CSV file into a DataFrame and Embeddings stored as str to lists
    df = pd.read_csv(csv_file)
    df["embedding"] = df["embedding"].apply(lambda x: list(map(float, x.split(","))))

    # Compute cosine similarity between the query vector and each vector in the 'embedding' column
    df["similarity"] = df["embedding"].apply(
        lambda x: cosine_similarity([x], [query_vector])[0][0]
    )

    if threshold is not None:
        # Filter the DataFrame based on the threshold
        results = df[df["similarity"] >= threshold]
    else:
        # Sort by similarity in descending order and get the top n
        results = df.nlargest(top_n, "similarity")

    # Return the image paths corresponding to the top N similar vectors or vectors with similarity higher than threshold
    top_img_paths = results["image_path"].to_list()

    return top_img_paths


def cosine_similarity_search_faiss(
    query_vector, database_filepath, top_n=None, threshold=None
):
    # Ensure at least one of top_n or threshold is set
    if top_n is None and threshold is None:
        raise ValueError("Either top_n or threshold must be specified.")

    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)
    query_vector = query_vector.astype("float32")
    faiss_path = database_filepath.replace(".csv", ".bin")

    # Load the faiss index
    index = faiss.read_index(faiss_path)

    if top_n:
        distances, indices = index.search(query_vector.reshape(1, -1), top_n)
        indices = indices[0]
    else:
        # Perform a Range Search
        no_of_beighbours, distances, indices = index.range_search(
            query_vector.reshape(1, -1), threshold
        )
        indices = indices.tolist()

    df = pd.read_csv(database_filepath)

    top_img_paths = [df["image_path"].iloc[i - 1] for i in indices if i != -1]

    return top_img_paths
