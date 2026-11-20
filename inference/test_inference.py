"""运行：python -m unittest inference.test_inference。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from inference.cache import Catalog
from model_management.store import ModelStore
from redis_service import RedisService

class MemoryRedis:
    value = None
    def get(self, key):
        return self.value
    def set(self, key, value, ex=None):
        self.value = value

class InferenceTest(unittest.TestCase):
    def test_probability_chart(self):
        import math
        import xml.etree.ElementTree as ET
        from train.api import app
        for count in (1, 12, 50):
            items = [{"class_name": f"标签<{index}>", "probability": .8 if index == 0 else .2 / count}
                     for index in range(count)]
            response = TestClient(app).post("/inference/chart", json={"items": items})
            self.assertEqual(response.status_code, 200)
            self.assertIn("image/svg+xml", response.headers["content-type"])
            svg = ET.fromstring(response.text)
            circles = svg.findall(".//{http://www.w3.org/2000/svg}circle[@data-node-body]")
            self.assertEqual(len(circles), count)
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}circle[@class='node-rim']")), count)
            self.assertTrue(all(circle.get("filter") is None for circle in circles))
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}line")), count * (count - 1) // 2)
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}linearGradient")), count * (count - 1) // 2)
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}radialGradient")), count)
            self.assertTrue(all(circle.get("fill").startswith("hsl(") for circle in circles))
            self.assertEqual(len(svg.findall(".//{http://www.w3.org/2000/svg}ellipse")), 0)
            for line in svg.findall(".//{http://www.w3.org/2000/svg}line"):
                self.assertGreaterEqual(float(line.get("stroke-width")), 1.8)
            self.assertIn("标签&lt;0&gt;", response.text)
            previous = 0
            for index, circle in enumerate(circles):
                x, y, radius = (float(circle.get(key)) for key in ("cx", "cy", "r"))
                distance = math.hypot(x, y)
                self.assertGreaterEqual(distance + .001, previous)
                previous = distance
                for other in circles[:index]:
                    self.assertGreaterEqual(math.hypot(x - float(other.get("cx")), y - float(other.get("cy"))),
                                            radius + float(other.get("r")))
        for items in ([], [{"class_name": "非法", "probability": 2}]):
            self.assertEqual(TestClient(app).post("/inference/chart", json={"items": items}).status_code, 422)

    def test_project_lifecycle(self):
        from train.api import app
        with patch("train.api.catalog") as catalog:
            with TestClient(app):
                catalog.start.assert_called_once_with()
                catalog.stop.assert_not_called()
            catalog.stop.assert_called_once_with()

    def test_legacy_server_lifecycle(self):
        from app import app, run_server
        for error in (None, KeyboardInterrupt(), RuntimeError("server failed")):
            with patch("redis_service.redis_service") as redis, patch.object(app, "run", side_effect=error):
                if error is None:
                    run_server()
                else:
                    with self.assertRaises(type(error)):
                        run_server()
                redis.start.assert_called_once_with()
                redis.stop.assert_called_once_with()

    def test_redis_package_operations(self):
        store = Mock()
        store.inference_catalog.return_value = [{"id": "model1", "labels": {"0": "体育"}}]
        redis = Mock()
        cache = Catalog(store=store, cache=redis)
        cache.start()
        redis.start.assert_called_once_with()
        redis.set_json.assert_called_once_with(cache.key, store.inference_catalog.return_value, ttl=300)
        redis.get_json.return_value = store.inference_catalog.return_value
        self.assertEqual(cache.get(), store.inference_catalog.return_value)
        redis.get_json.assert_called_once_with(cache.key)
        store.inference_catalog.assert_called_once_with()
        for value in (None, {}, "invalid"):
            redis.get_json.return_value = value
            self.assertEqual(cache.get(), store.inference_catalog.return_value)
        redis.get_json.side_effect = RuntimeError("unavailable")
        redis.set_json.side_effect = RuntimeError("unavailable")
        self.assertEqual(cache.get(), store.inference_catalog.return_value)
        cache.stop()
        redis.stop.assert_called_once_with()

    def test_start_failure_closes_redis(self):
        store, redis = Mock(), Mock()
        store.inference_catalog.side_effect = RuntimeError("database unavailable")
        with self.assertRaises(RuntimeError):
            Catalog(store=store, cache=redis).start()
        redis.stop.assert_called_once_with()

    def test_shutdown_local_redis(self):
        from redis_service.local import stop_local_redis
        from redis.exceptions import ConnectionError
        client = Mock()
        client.connection_pool.connection_kwargs = {"host": "127.0.0.1"}
        stop_local_redis(client)
        client.shutdown.assert_called_once_with(nosave=True)
        client.close.assert_called_once()
        client.reset_mock()
        client.shutdown.side_effect = ConnectionError("already stopped")
        stop_local_redis(client)
        client.close.assert_called_once()
        client.reset_mock()
        client.connection_pool.connection_kwargs = {"host": "remote"}
        stop_local_redis(client)
        client.shutdown.assert_not_called()

    def test_cache_and_prediction(self):
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from inference.service import predict
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ModelStore(root / "registry.sqlite3")
            for index in range(2):
                path = root / f"model{index}"
                path.mkdir()
                job = SimpleNamespace(id=str(index), model="rf", purpose="news", finished_at=f"2026-11-2{index}", source="train", test_source="test")
                store.register(job=job, model_path=path, params={}, evaluation={}, label_mapping={"0":"体育", "1":"财经"})
            client = MemoryRedis()
            cache = Catalog(store, RedisService(client=client))
            items = cache.get()
            self.assertEqual([item["id"] for item in items], ["1", "0"])
            with patch.object(store, "inference_catalog", side_effect=AssertionError("cache miss")):
                self.assertEqual(cache.get(), items)
            client.value = "invalid json"
            self.assertEqual(cache.get(), items)
            with patch.object(client, "get", side_effect=ConnectionError()), patch.object(client, "set", side_effect=ConnectionError()):
                self.assertEqual(cache.get(), items)
            record = items[0]
            path = Path(record["path"])
            vectorizer = TfidfVectorizer().fit(["足球 比赛", "股票 金融"])
            model = RandomForestClassifier(n_estimators=10, bootstrap=False, random_state=42).fit(vectorizer.transform(["足球 比赛", "股票 金融"]), [0,1])
            joblib.dump(vectorizer, path / "tfidf_vectorizer.joblib")
            joblib.dump(model, path / "random_forest.joblib")
            prediction = predict(record, "足球 比赛")
            self.assertEqual(prediction["class_name"], "体育")
            self.assertEqual({item["label"] for item in prediction["probabilities"]}, {"0", "1"})
            self.assertAlmostEqual(sum(item["probability"] for item in prediction["probabilities"]), 1.0)
            self.assertEqual(max(prediction["probabilities"], key=lambda item: item["probability"])["label"], prediction["label"])
            from train.api import app
            with patch("inference.api.catalog", cache):
                http = TestClient(app)
                self.assertEqual(http.get("/inference/models").status_code, 200)
                self.assertNotIn("path", http.get("/inference/models").json()["items"][0])
                payload = {"model_id":"1", "purpose":"news", "text":"足球 比赛"}
                self.assertEqual(http.post("/inference/predict", json=payload).json()["class_name"], "体育")
                for key, value, status in [("text", " ", 422), ("model_id", "missing", 404), ("purpose", "intent", 400)]:
                    self.assertEqual(http.post("/inference/predict", json={**payload,key:value}).status_code, status)
                self.assertEqual(http.get("/inference").status_code, 200)
            store.update("1", {"name":"renamed"})
            self.assertEqual(cache.refresh()[0]["name"], "renamed")
            store.delete("1")
            self.assertEqual([item["id"] for item in cache.refresh()], ["0"])

if __name__ == "__main__":
    unittest.main()


