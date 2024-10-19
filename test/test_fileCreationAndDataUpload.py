import os
import unittest
from unittest.mock import patch

from src.facematch.database_functions import upload_embedding_to_database


class TestUploadEmbeddingToDatabase(unittest.TestCase):
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_upload_embedding_to_database(self, mock_open):

        # Sample data for testing
        data = [
            {"name": "Alice", "age": 30, "city": "New York"},
            {"name": "Bob", "age": 25, "city": "San Francisco"},
        ]

        # Call the function to test
        upload_embedding_to_database(data, "data/test_embeddings.csv")

        # Assert that open was called with the correct file path and mode
        mock_open.assert_called_once_with(
            "data/test_embeddings.csv", mode="w", newline=""
        )

        # Get the handle to the mock file
        handle = mock_open()

        # Get the written content
        written_content = handle.write.call_args_list

        # Extracting the arguments to verify content written
        written_lines = [args[0][0] for args in written_content]

        # Assert that the header and data are written correctly
        self.assertIn("name,age,city\r\n", written_lines)
        self.assertIn("Alice,30,New York\r\n", written_lines)
        self.assertIn("Bob,25,San Francisco\r\n", written_lines)

    def test_directory_creation(self):
        # Test to ensure the directory is created
        csv_file = "data/test_embeddings.csv"

        # Remove the directory if it exists for the test
        if os.path.exists(csv_file):
            os.remove(csv_file)

        # Call the function
        data = [{"name": "Test", "age": 0, "city": "Test City"}]
        upload_embedding_to_database(data, csv_file)

        # Assert that the directory now exists
        self.assertTrue(os.path.exists(csv_file))

        # Cleanup
        if os.path.exists(csv_file):
            os.remove(csv_file)


if __name__ == "__main__":
    unittest.main()
