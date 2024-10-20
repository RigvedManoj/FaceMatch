import os

import pandas as pd


def upload_embedding_to_database(data, database_filepath):
    csv_file = database_filepath
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    df = pd.DataFrame(data)
    df["embedding"] = df["embedding"].apply(lambda x: ",".join(map(str, x)))
    df["bbox"] = df["bbox"].apply(lambda x: ",".join(map(str, x)))

    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", index=False, header=False)
    else:
        df.to_csv(csv_file, index=False)
