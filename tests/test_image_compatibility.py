import io
import unittest
from PIL import Image

class ImageCompatibilityTests(unittest.TestCase):
    def test_image_load_convert_thumbnail_and_preview_resize(self):
        original = Image.new('RGB', (260, 260), color=(30, 41, 59))
        encoded = io.BytesIO()
        original.save(encoded, format='PNG')
        encoded.seek(0)
        loaded = Image.open(encoded).convert('RGB')
        self.assertEqual(loaded.getpixel((0, 0)), (30, 41, 59))
        loaded.thumbnail((130, 130), Image.LANCZOS)
        self.assertEqual(loaded.size, (130, 130))
        self.assertEqual(loaded.resize((760, 520), Image.LANCZOS).size, (760, 520))

if __name__ == '__main__':
    unittest.main()
