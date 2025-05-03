import unittest
import os
import tempfile
from data.raw.download import download_cifar, save_dataset

class TestCIFARDownload(unittest.TestCase):
    def setUp(self):
        self.test_save_path = tempfile.mkdtemp()

    def test_download_cifar10(self):

        try:
            (x_train, y_train), (x_test, y_test) = download_cifar(save_path=self.test_save_path)
            
            self.assertIsNotNone(x_train)
            self.assertIsNotNone(y_train)
            self.assertIsNotNone(x_test)
            self.assertIsNotNone(y_test)
        except Exception as e:
            self.fail(f"download_cifar() raised {type(e).__name__} unexpectedly!")

    def test_dataset_dimensions(self):

        (x_train, y_train), (x_test, y_test) = download_cifar(save_path=self.test_save_path)
        
        self.assertEqual(x_train.shape[0], y_train.shape[0], "Training images and labels must match")
        self.assertTrue(0 < len(x_train) <= 50000, "Training dataset size is incorrect")
        
        self.assertEqual(x_test.shape[0], y_test.shape[0], "Test images and labels must match")
        self.assertTrue(0 < len(x_test) <= 10000, "Test dataset size is incorrect")
        
        self.assertEqual(x_train.shape[1:], (32, 32, 3), "Image dimensions should be 32x32x3")

    def test_save_dataset(self):
        """Check if dataset can be saved correctly"""
        dataset = download_cifar(save_path=self.test_save_path)
        
        save_dataset(dataset, save_path=self.test_save_path)

        expected_files = [
            'x_train.npy', 'y_train.npy', 
            'x_test.npy', 'y_test.npy'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.test_save_path, filename)
            self.assertTrue(os.path.exists(filepath), f"File {filename} was not created")

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError, msg="Invalid dataset name should raise ValueError"):
            download_cifar(dataset_name='invalid_dataset')

    def tearDown(self):
        """Clean up temporary files after tests"""
        for root, dirs, files in os.walk(self.test_save_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.test_save_path)

if __name__ == '__main__':
    unittest.main()