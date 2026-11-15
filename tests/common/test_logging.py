"""setup_logging honours LOG_LEVEL from the environment.

Compose passes LOG_LEVEL into every container (the dev stack sets DEBUG); the
value is documented in `.env.example` and the README, so it must actually reach
the logger: environment > configs/config.yaml > INFO.
"""
import logging
import os
import unittest
from unittest import mock


class LogLevelTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("LOG_LEVEL", None)
        self.addCleanup(os.environ.pop, "LOG_LEVEL", None)

    def level(self, *, env=None, yaml="INFO"):
        def fake_config(key, default=None):
            return {"logging.level": yaml,
                    "logging.format": "%(levelname)s %(message)s"}.get(key, default)

        with mock.patch.dict(os.environ, {"LOG_LEVEL": env} if env is not None else {}):
            with mock.patch("common.config.get_config", side_effect=fake_config):
                from common.logging_config import setup_logging
                logger = setup_logging("test_log_level", log_to_file=False)
                return logger.getEffectiveLevel()

    def test_env_overrides_yaml(self):
        # The dev stack sets LOG_LEVEL=DEBUG; that must actually take effect.
        self.assertEqual(self.level(env="DEBUG", yaml="INFO"), logging.DEBUG)

    def test_yaml_used_when_env_unset(self):
        self.assertEqual(self.level(env=None, yaml="WARNING"), logging.WARNING)

    def test_invalid_value_falls_back_to_info(self):
        self.assertEqual(self.level(env="NOT_A_LEVEL", yaml="DEBUG"), logging.INFO)

    def test_lowercase_value_is_accepted(self):
        self.assertEqual(self.level(env="debug", yaml="INFO"), logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
