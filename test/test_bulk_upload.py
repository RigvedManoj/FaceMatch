import os
import unittest

from src.facematch.interface import FaceMatchModel
from src.facematch.resource_path import get_resource_path


class TestBulkUpload(unittest.TestCase):

    def setUp(self):
        self.image_directory_path = get_resource_path("sample_images")
        self.database_path = get_resource_path("test_db.csv")

    def test_bulk_upload_success(self):
        face_match_object = FaceMatchModel()
        result = face_match_object.bulk_upload(
            self.image_directory_path, self.database_path
        )
        self.assertEqual("Successfully uploaded to database", result)

    def tearDown(self):
        if os.path.exists(self.database_path):
            os.remove(self.database_path)


if __name__ == "__main__":
    unittest.main()
