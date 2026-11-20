"""运行：python -m unittest redis_service.test_service。"""
import unittest
from unittest.mock import Mock, patch

from redis_service import RedisService


class RedisServiceTest(unittest.TestCase):
    def test_json_and_lifecycle(self):
        client = Mock()
        service = RedisService(client=client)
        service.set_json("catalog", ["体育"], ttl=300)
        client.set.assert_called_once_with("catalog", '["体育"]', ex=300)
        client.get.return_value = '["体育"]'
        self.assertEqual(service.get_json("catalog"), ["体育"])
        client.get.return_value = None
        self.assertIsNone(service.get_json("missing"))
        client.get.return_value = "invalid json"
        with self.assertRaises(ValueError):
            service.get_json("catalog")
        with patch("redis_service.ensure_local_redis") as start, patch("redis_service.stop_local_redis") as stop:
            service.start()
            service.stop()
            start.assert_called_once_with(client)
            stop.assert_called_once_with(client)


if __name__ == "__main__":
    unittest.main()
