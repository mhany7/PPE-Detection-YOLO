import unittest
from src.inference import PPEDetector
from PIL import Image

class TestPPEDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PPEDetector(pytorch_model_path="models/model.pt")

    def test_pytorch_inference(self):
        img = Image.open("data/sample_images/sample 1.png")
        result = self.detector.detect(img, model_type="pytorch")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()