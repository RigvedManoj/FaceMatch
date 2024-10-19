import os
import unittest

from src.facematch import face_representation


class TestApp(unittest.TestCase):

    def test_face_representation(self):
        current_dir = os.path.dirname(__file__)
        image_path = os.path.join(
            current_dir, "..", "resources", "sample_images/single.jpg"
        )
        result = face_representation.detect_faces_and_get_embeddings(image_path)
        assert len(result[0]["embedding"]) == 512
        assert result[0]["region"] == (222, 200, 90, 120)


if __name__ == "__main__":
    unittest.main()
