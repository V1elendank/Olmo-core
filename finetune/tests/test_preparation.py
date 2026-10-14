"""Model-preparation failure and concurrency checks without loading any model weights."""
import concurrent.futures
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftlib.config import DatasetCfg, ModelCfg, RunConfig, TrainCfg
from ftlib.models import is_prepared, prepare_model
from ftlib.status import models


class PreparationTests(unittest.TestCase):
    def test_failed_validation_is_retried_and_concurrent_requests_convert_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = RunConfig(
                name="test",
                model=ModelCfg("hf_local", str(root / "input"), "olmo3_7b"),
                tokenizer="test",
                data=[DatasetCfg("test", "unused")],
                train=TrainCfg(steps=1),
            )
            with patch.dict("os.environ", {"OLMO_ROOT": str(root)}):
                output = cfg.model_dir

                def fail_validation(_source, _arch, out, _validate):
                    (out / "model_and_optim").mkdir(exist_ok=True)
                    (out / "model_and_optim" / ".metadata").touch()
                    raise RuntimeError("logit validation failed")

                with patch("ftlib.models._convert_hf", side_effect=fail_validation):
                    with self.assertRaisesRegex(RuntimeError, "logit validation"):
                        prepare_model(cfg)
                self.assertFalse(is_prepared(output))
                self.assertTrue((output / ".ft-preparing").exists())
                self.assertEqual(models(root), [])

                entered, finish = threading.Event(), threading.Event()

                def convert(*_):
                    entered.set()
                    if not finish.wait(5):
                        raise TimeoutError("test converter was not released")

                with patch("ftlib.models._convert_hf", side_effect=convert) as converter:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        first = pool.submit(prepare_model, cfg)
                        try:
                            self.assertTrue(entered.wait(5))
                            second = pool.submit(prepare_model, cfg)
                            self.assertFalse(is_prepared(output))
                        finally:
                            finish.set()
                        self.assertEqual(first.result(timeout=5), output)
                        self.assertEqual(second.result(timeout=5), output)
                    self.assertEqual(converter.call_count, 1)
                self.assertTrue(is_prepared(output))
                self.assertEqual(len(models(root)), 1)
                self.assertTrue((output / "ft_source.json").is_file())
                self.assertFalse((output / ".ft-preparing").exists())

    def test_local_core_checkpoint_is_used_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            (checkpoint / "model_and_optim").mkdir()
            (checkpoint / "model_and_optim" / ".metadata").touch()
            cfg = RunConfig(
                name="test",
                model=ModelCfg("olmo_core", folder),
                tokenizer="test",
                data=[DatasetCfg("test", "unused")],
                train=TrainCfg(steps=1),
            )
            self.assertEqual(prepare_model(cfg), checkpoint)
            self.assertFalse((checkpoint / ".ft-preparing").exists())
