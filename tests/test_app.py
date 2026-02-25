import unittest
from deployment.app import app

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()


    def test_home(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MLops Deployment with FastAPI', response.data)

    def test_predict(self):
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'predicted_class', response.data)

if __name__ == '__main__':
    unittest.main()