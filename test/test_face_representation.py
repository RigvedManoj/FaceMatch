import unittest

from src.facematch.face_representation import detect_faces_and_get_embeddings
from src.facematch.resource_path import get_resource_path


class TestApp(unittest.TestCase):

    def test_face_representation(self):
        image_path = get_resource_path("sample_images/single.jpg")
        result = detect_faces_and_get_embeddings(image_path)
        assert len(result[0]["embedding"]) == 512
        assert result[0]["bbox"] == [222, 200, 90, 120]


if __name__ == "__main__":
    unittest.main()
