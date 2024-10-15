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
    df["Embeddings"] = df["Embeddings"].apply(lambda x: list(map(float, x.split(","))))

    # Compute cosine similarity between the query vector and each vector in the 'embedding' column
    df["similarity"] = df["Embeddings"].apply(
        lambda x: cosine_similarity([x], [query_vector])[0][0]
    )

    if threshold is not None:
        # Filter the DataFrame based on the threshold
        results = df[df["similarity"] >= threshold]
    else:
        # Sort by similarity in descending order and get the top n
        results = df.nlargest(top_n, "similarity")

    # Return the image paths corresponding to the top N similar vectors or vectors with similarity higher than threshold
    top_img_paths = results["img_path"].to_list()

    return top_img_paths
