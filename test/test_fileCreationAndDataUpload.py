import os
import unittest

import pandas as pd

from src.facematch.database_functions import upload_embedding_to_database


class TestUploadEmbeddingToDatabase(unittest.TestCase):
    def test_upload_embedding_to_database(self):

        # Sample data for testing
        test_data = [
            {
                "image_path": "test/data/img1.png",
                "embedding": [1, 2, 3],
                "bbox": [1, 2, 300, 200],
            },
            {
                "image_path": "test/data/img2.png",
                "embedding": [4, 2, 3],
                "bbox": [5, 2, 300, 200],
            },
        ]

        # Database file
        self.csv_file = "test/data/test_embeddings.csv"

        # Upload data
        upload_embedding_to_database(test_data, self.csv_file)

        # Read data
        df = pd.read_csv(self.csv_file)

        # Assert that the header and data are written correctly
        self.assertEqual(df.iloc[0, 0], "test/data/img1.png")
        self.assertEqual(df.iloc[0, 1], "1,2,3")
        self.assertEqual(df.iloc[0, 2], "1,2,300,200")

    def test_directory_creation(self):

        self.csv_file = "test/data/test_embeddings.csv"

        # Remove the directory if it exists for the test
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)

        # Call the function
        data = [
            {
                "image_path": "test/data/img1.png",
                "embedding": [1, 2, 3],
                "bbox": (1, 2, 300, 200),
            }
        ]
        upload_embedding_to_database(data, self.csv_file)

        # Assert that the directory now exists
        self.assertTrue(os.path.exists(self.csv_file))

    def tearDown(self):
        # Clean up the temporary CSV file
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)


if __name__ == "__main__":
    unittest.main()
