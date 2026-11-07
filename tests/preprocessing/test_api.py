"""Tests for preprocessing API endpoints.

Run with: python -m unittest tests.preprocessing.test_api
"""

import unittest
try:
    from fastapi.testclient import TestClient
    from services.preprocessing.api.main import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not available")
class TestHealthEndpoint(unittest.TestCase):
    """Test health check endpoint."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not available")
class TestPreprocessEndpoint(unittest.TestCase):
    """Test single text preprocessing endpoint."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_basic_preprocess(self):
        response = self.client.post(
            "/preprocess",
            json={"text": "The quick brown fox"}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("original", data)
        self.assertIsInstance(data["tokens"], list)
    
    def test_invalid_config_key_returns_422(self):
        response = self.client.post(
            "/preprocess",
            json={
                "text": "Test",
                "config": {"invalid_key": "value"}
            }
        )
        
        self.assertEqual(response.status_code, 422)
        self.assertIn("Invalid configuration keys", response.json()["detail"])
    
    def test_text_too_long_rejected(self):
        long_text = "a" * 100001
        response = self.client.post(
            "/preprocess",
            json={"text": long_text}
        )
        
        self.assertEqual(response.status_code, 422)
    
    def test_hyphenated_preserved(self):
        response = self.client.post(
            "/preprocess",
            json={
                "text": "state-of-the-art machine-learning",
                "config": {
                    "remove_punctuation": True,
                    "remove_stopwords": False
                }
            }
        )
        
        self.assertEqual(response.status_code, 200)
        tokens = response.json()["tokens"]
        
        # Check that hyphenated words exist
        has_state = any("state-of-the-art" in t.lower() for t in tokens)
        has_ml = any("machine-learning" in t.lower() for t in tokens)
        self.assertTrue(has_state or has_ml, f"Tokens: {tokens}")


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not available")
class TestBatchEndpoint(unittest.TestCase):
    """Test batch preprocessing endpoint."""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_batch_basic(self):
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": ["First doc", "Second doc", "Third doc"]}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("results", data)
        self.assertIn("total", data)
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["results"]), 3)
    
    def test_batch_results_are_token_lists(self):
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": ["Doc one", "Doc two"]}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Each result should be a list
        for result in data["results"]:
            self.assertIsInstance(result, list)
    
    def test_too_many_texts_rejected(self):
        texts = ["text"] * 1001
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": texts}
        )
        
        self.assertEqual(response.status_code, 422)
    
    def test_text_too_long_in_batch(self):
        texts = ["normal", "a" * 100001, "normal"]
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": texts}
        )
        
        self.assertEqual(response.status_code, 422)
        self.assertIn("exceeds maximum length", response.json()["detail"])
    
    def test_empty_list(self):
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": []}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 0)
        self.assertEqual(data["results"], [])
    
    def test_batch_hyphenated(self):
        response = self.client.post(
            "/preprocess/batch",
            json={
                "texts": ["state-of-the-art systems", "many machine-learning models"],
                "config": {
                    "remove_stopwords": False,
                    "use_lemmatization": False
                }
            }
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("state-of-the-art", data["results"][0])
        self.assertIn("machine-learning", data["results"][1])
    
    def test_batch_blank_texts(self):
        """Blanks stay aligned as empty token lists instead of failing or mixing up order."""
        response = self.client.post(
            "/preprocess/batch",
            json={"texts": ["hello", "", "world systems"]}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(data["results"][0], ["hello"])
        self.assertEqual(data["results"][1], [])
        # default config lemmatizes (systems -> system)
        self.assertEqual(data["results"][2], ["world", "system"])


if __name__ == "__main__":
    unittest.main()
