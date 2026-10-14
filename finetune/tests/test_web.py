"""CPU-only HTTP and launch regression tests; never call real Slurm or load a model."""
import hashlib
import http.client
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftlib.config import DatasetCfg, ModelCfg, RunConfig, TrainCfg, config_paths, load_config
from ftlib.store import Store
from ftlib.web import Application, _worker, make_server


def model_files():
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return {
        "config.json": json.dumps(
            {"model_type": "olmo3", "hidden_size": 4096, "num_hidden_layers": 32}
        ).encode(),
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": b'{"chat_template":"example"}',
        "model.safetensors": struct.pack("<Q", len(header)) + header + b"\0\0\0\0",
    }


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"OLMO_ROOT": str(self.root)})
        self.env.start()
        self.spawned = []
        self.app = Application(self.root, spawn=lambda cfg, path: self.spawned.append((cfg, path)))
        self.server = make_server(self.app, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.env.stop()
        self.temp.cleanup()

    def request(self, method, path, body=None, headers=None):
        content = json.dumps(body).encode() if isinstance(body, dict) else body
        hdr = {"X-FT-Token": self.app.token, "Origin": f"http://127.0.0.1:{self.port}"}
        hdr.update(headers or {})
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request(method, path, body=content, headers=hdr)
        response = conn.getresponse()
        data = response.read()
        code = response.status
        conn.close()
        return code, json.loads(data) if data.startswith(b"{") else data

    def upload(self, kind, files):
        code, data = self.request(
            "POST",
            "/api/uploads",
            {
                "kind": kind,
                "name": "My upload",
                "files": [{"path": p, "size": len(b)} for p, b in files.items()],
            },
        )
        self.assertEqual(code, 201, data)
        asset = data["asset"]
        for i, content in enumerate(files.values()):
            code, data = self.request(
                "PUT", f'/api/uploads/{asset["id"]}/files/{i}', content, {"X-Upload-Offset": "0"}
            )
            self.assertEqual(code, 200, data)
        code, data = self.request("POST", f'/api/uploads/{asset["id"]}/complete', {})
        self.assertEqual(code, 200, data)
        return data["asset"]

    def inputs(self):
        model = self.upload("model", model_files())
        data = self.upload("dataset", {"data.jsonl": b'{"text":"hello"}\n'})
        return {
            "request_id": "a" * 32,
            "name": "browser-run",
            "model_id": model["id"],
            "dataset_id": data["id"],
            "arch": "olmo3_7b",
            "format": "text",
        }

    def test_upload_selection_start_preserves_fit_gate_and_local_sources(self):
        body = self.inputs()
        with patch("ftlib.web.shutil.which", return_value="/fake/sbatch"):
            code, response = self.request("POST", "/api/runs", body)
        self.assertEqual(code, 202, response)
        self.assertEqual(len(self.spawned), 1)
        config, record = self.spawned[0]
        cfg = load_config(config)
        self.assertEqual(cfg.model.source, "hf_local")
        self.assertTrue(Path(cfg.model.path).is_dir())
        self.assertTrue(Path(cfg.data[0].source).is_dir())
        self.assertEqual(cfg.tokenizer, cfg.model.path)
        self.assertEqual(cfg.slurm.fit_check_steps, 5)
        self.assertTrue(cfg.export.hf)
        self.assertIn(config, config_paths())
        with patch("ftlib.web.subprocess.run") as run:
            run.return_value.returncode = 0
            _worker(config, record)
            command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "submit.sh")
        self.assertNotIn("--no-fit-check", command)

    def test_duplicate_click_is_idempotent_and_new_request_cannot_resume_by_name(self):
        body = self.inputs()
        with patch("ftlib.web.shutil.which", return_value="/fake/sbatch"):
            self.assertEqual(self.request("POST", "/api/runs", body)[0], 202)
            self.assertEqual(self.request("POST", "/api/runs", body)[0], 202)
            self.assertEqual(len(self.spawned), 1)
            body["request_id"] = "b" * 32
            self.assertEqual(self.request("POST", "/api/runs", body)[0], 400)

    def test_no_slurm_disables_launch(self):
        body = self.inputs()
        with patch("ftlib.web.shutil.which", return_value=None):
            self.assertFalse(self.request("GET", "/api/session")[1]["slurm_available"])
            self.assertEqual(self.request("POST", "/api/runs", body)[0], 400)
        self.assertFalse(self.spawned)

    def test_second_panel_cannot_race_with_uploads(self):
        with self.assertRaisesRegex(ValueError, "Another panel"):
            Application(self.root)

    def test_malformed_model_config_returns_readable_error(self):
        store = self.app.store
        record = store.create("model", "bad config", [{"path": "config.json", "size": 2}])
        store.chunk(record["id"], 0, 0, b"[]")
        code, reply = self.request("POST", f'/api/uploads/{record["id"]}/complete', {})
        self.assertEqual(code, 400)
        self.assertIn("JSON object", reply["error"])

    def test_token_origin_host_and_traversal_are_rejected(self):
        for headers in [
            {"X-FT-Token": ""},
            {"Origin": "https://evil.example"},
            {"Host": "evil.example"},
        ]:
            self.assertEqual(self.request("POST", "/api/uploads", {}, headers)[0], 403)
        for filename in [
            "../x.json",
            "/x.json",
            "a/../../x.json",
            "a\\x.json",
            ".hidden.json",
            "a//x.json",
        ]:
            self.assertEqual(
                self.request(
                    "POST",
                    "/api/uploads",
                    {"kind": "dataset", "name": "bad", "files": [{"path": filename, "size": 10}]},
                )[0],
                400,
            )

    def test_interrupted_upload_resumes_at_exact_offset(self):
        record = self.app.store.create("dataset", "resume", [{"path": "data.jsonl", "size": 6}])
        asset_id = record["id"]
        self.assertEqual(
            self.request(
                "PUT", f"/api/uploads/{asset_id}/files/0", b"abc", {"X-Upload-Offset": "0"}
            )[0],
            200,
        )
        # A fresh Store instance represents a restarted server.
        self.assertEqual(Store(self.root).offsets(asset_id)[0]["offset"], 3)
        self.assertEqual(self.request("POST", f"/api/uploads/{asset_id}/complete", {})[0], 400)
        self.assertEqual(
            self.request(
                "PUT", f"/api/uploads/{asset_id}/files/0", b"abc", {"X-Upload-Offset": "0"}
            )[0],
            400,
        )
        self.assertEqual(
            self.request(
                "PUT", f"/api/uploads/{asset_id}/files/0", b"def", {"X-Upload-Offset": "3"}
            )[0],
            200,
        )
        done = self.request("POST", f"/api/uploads/{asset_id}/complete", {})[1]["asset"]
        self.assertEqual(done["hashes"][0]["sha256"], hashlib.sha256(b"abcdef").hexdigest())
        self.assertEqual(
            self.request("PUT", f"/api/uploads/{asset_id}/files/0", b"x", {"X-Upload-Offset": "6"})[
                0
            ],
            400,
        )

    def test_incomplete_model_is_not_selectable(self):
        store = self.app.store
        r = store.create("model", "bad", [{"path": "config.json", "size": 2}])
        store.chunk(r["id"], 0, 0, b"{}")
        self.assertEqual(self.request("POST", f'/api/uploads/{r["id"]}/complete', {})[0], 400)
        self.assertEqual(store.get(r["id"])["state"], "uploading")

    def test_missing_model_shard_is_rejected(self):
        files = model_files()
        files["model.safetensors.index.json"] = b'{"weight_map":{"x":"missing.safetensors"}}'
        r = self.app.store.create(
            "model", "bad shards", [{"path": p, "size": len(v)} for p, v in files.items()]
        )
        for i, v in enumerate(files.values()):
            self.app.store.chunk(r["id"], i, 0, v)
        with self.assertRaisesRegex(ValueError, "shard"):
            self.app.store.finalize(r["id"])

    def test_remove_unused_but_protect_referenced_inputs(self):
        unused = self.upload("dataset", {"unused.jsonl": b'{"text":"unused"}\n'})
        self.assertEqual(self.request("DELETE", "/api/uploads/" + unused["id"])[0], 200)
        self.assertFalse(self.app.store.directory(unused["id"]).exists())
        body = self.inputs()
        with patch("ftlib.web.shutil.which", return_value="/fake/sbatch"):
            self.app.start(body)
        for asset_id in [body["model_id"], body["dataset_id"]]:
            self.assertEqual(self.request("DELETE", "/api/uploads/" + asset_id)[0], 400)
            self.assertTrue(self.app.store.directory(asset_id).exists())

    def test_unsupported_gpu_count_and_wrong_model_family_do_not_submit(self):
        body = self.inputs()
        with patch("ftlib.web.shutil.which", return_value="/fake/sbatch"):
            for changes in [
                {"gpus": 4},
                {"arch": "olmo2_7b"},
                {"global_batch_seqs": 3},
                {"lr": -1},
            ]:
                self.assertEqual(self.request("POST", "/api/runs", dict(body, **changes))[0], 400)
        self.assertFalse(self.spawned)

    def test_changed_upload_versions_and_external_files_invalidate_caches(self):
        a = self.upload("dataset", {"same.jsonl": b'{"text":"one"}\n'})
        b = self.upload("dataset", {"same.jsonl": b'{"text":"two"}\n'})
        self.assertNotEqual(a["id"], b["id"])
        self.assertNotEqual(a["digest"], b["digest"])
        model = self.root / "external-model"
        model.mkdir()
        weights = model / "weights.safetensors"
        data = self.root / "external.jsonl"
        weights.write_bytes(b"old")
        data.write_bytes(b"old")
        cfg = RunConfig(
            name="same-run",
            model=ModelCfg(source="hf_local", path=str(model), arch="olmo3_7b"),
            tokenizer=str(model),
            data=[DatasetCfg(name="d", source=str(data), format="text")],
            train=TrainCfg(steps=1),
        )
        old_model, old_data = cfg.model_key, cfg.data_key
        weights.write_bytes(b"new model")
        data.write_bytes(b"new data")
        self.assertNotEqual(old_model, cfg.model_key)
        self.assertNotEqual(old_data, cfg.data_key)

    def test_launch_failure_is_visible_and_input_references_retained(self):
        body = self.inputs()

        def fail(*_):
            raise OSError("test spawn failure")

        self.app.spawn = fail
        with patch("ftlib.web.shutil.which", return_value="/fake/sbatch"):
            result = self.app.start(body)
        self.assertEqual(result["state"], "failed")
        self.assertIn("test spawn failure", self.app.launches()[0]["error"])

    def fake_slurm(self, ready=True, fail_train=False, name="chain"):
        """Supply fake cluster commands to exercise real submission code offline."""
        fakebin = self.root / "bin"
        fakebin.mkdir()
        run_dir = self.root / "runs" / name
        model = self.root / "ready-model" / "model_and_optim"
        model.mkdir(parents=True)
        if ready:
            (model / ".metadata").touch()
        data = self.root / "ready-data"
        data.mkdir()
        if ready:
            (data / "manifest.json").write_text("{}")
            (data / "pack.json").write_text("{}")
        variables = {
            "FT_EXPORT_HF": "1",
            "FT_FIT_STEPS": "5",
            "FT_MODEL_DIR": str(model.parent),
            "FT_DATA_DIR": str(data),
            "FT_RUN_DIR": str(run_dir),
            "FT_MAIL": "",
            "FT_PARTITION": "gpu",
            "FT_GPUS": "2",
            "FT_CPUS": "16",
            "FT_MEM": "240000",
            "FT_CONSTRAINT": "h100_94gb",
            "FT_NAME": name,
            "FT_TIME": "01:00:00",
        }
        import shlex

        exports = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in variables.items())
        (fakebin / "python").write_text(
            "#!/bin/sh\nif [ \"$2\" = paths ]; then\ncat <<'EOF'\n"
            + exports
            + "\nEOF\nelse echo OK; fi\n"
        )
        counter = self.root / "counter"
        args = self.root / "sbatch-args"
        (fakebin / "sbatch").write_text(
            f'#!/bin/sh\nn=$(cat "{counter}" 2>/dev/null || echo 100)\nn=$((n+1))\necho "$n" > "{counter}"\necho "$*" >> "{args}"\necho "$n"\n'
        )
        if fail_train:
            script = (fakebin / "sbatch").read_text()
            script = script.replace(
                'echo "$n"\n',
                'case "$*" in *ft-train-*) echo "test submission rejected" >&2; exit 1;; esac\necho "$n"\n',
            )
            (fakebin / "sbatch").write_text(script)
        (fakebin / "squeue").write_text("#!/bin/sh\nexit 0\n")
        for p in fakebin.iterdir():
            p.chmod(0o755)
        config = self.root / "chain.yaml"
        config.write_text("name: chain\n")
        env = dict(
            os.environ,
            PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
            OLMO_ROOT=str(self.root),
            OLMO_VENV=str(self.root / "missing-venv"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return env, config, args

    def test_real_submit_script_chains_fit_before_train(self):
        """Exercise the actual shell submitter without spending GPU time."""
        env, config, args = self.fake_slurm()
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["bash", str(repo / "finetune/scripts/submit.sh"), str(config)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = args.read_text().splitlines()
        self.assertEqual(len(calls), 3)
        self.assertIn("--fit-check", calls[0])
        self.assertIn("--dependency=afterok:101", calls[1])
        self.assertNotIn("--fit-check", calls[1])
        self.assertIn("--dependency=afterok:102", calls[2])

    def test_failed_conversion_is_not_treated_as_prepared_by_submitter(self):
        env, config, args = self.fake_slurm()
        (self.root / "ready-model" / ".ft-preparing").touch()
        repo = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["bash", str(repo / "finetune/scripts/submit.sh"), str(config)],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = args.read_text().splitlines()
        self.assertEqual(len(calls), 4)
        self.assertIn("ft-prep-chain", calls[0])
        self.assertIn("--dependency=afterok:101", calls[1])

    def run_detached_worker(self, fail_train=False):
        """Run the real HTTP -> detached worker -> shell -> fake Slurm path."""
        body = self.inputs()
        env, _, args = self.fake_slurm(ready=False, fail_train=fail_train, name=body["name"])
        processes = []

        def spawn(config, record):
            process = self.app._spawn(config, record)
            processes.append(process)
            return process

        self.app.spawn = spawn
        try:
            with patch.dict(os.environ, env):
                code, reply = self.request("POST", "/api/runs", body)
                self.assertEqual(code, 202, reply)
                self.assertEqual(len(processes), 1)
                self.assertEqual(os.getsid(processes[0].pid), processes[0].pid)
                processes[0].wait(timeout=15)
            record = self.app.launches()[0]
            return record, args.read_text().splitlines()
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_detached_worker_submits_all_four_stages(self):
        record, calls = self.run_detached_worker()
        self.assertEqual(record["state"], "submitted", record)
        self.assertEqual(
            [job["stage"] for job in record["jobs"]], ["prepare", "fit-check", "train", "export"]
        )
        self.assertEqual(len(calls), 4)
        for index, call in enumerate(calls[1:], 101):
            self.assertIn(f"--dependency=afterok:{index}", call)
        self.assertIn("--fit-check", calls[1])

    def test_detached_worker_reports_partial_submission_failure(self):
        record, calls = self.run_detached_worker(fail_train=True)
        self.assertEqual(record["state"], "failed", record)
        self.assertIn("test submission rejected", record["log"])
        self.assertEqual([job["stage"] for job in record["jobs"]], ["prepare", "fit-check"])
        self.assertEqual(len(calls), 3)

    def test_removed_upload_returns_404_for_browser_resume(self):
        asset = self.upload("dataset", {"data.jsonl": b"hello"})
        self.app.store.remove(asset["id"])
        self.assertEqual(self.request("GET", "/api/uploads/" + asset["id"])[0], 404)

    def test_malformed_shard_index_returns_readable_error(self):
        files = model_files()
        files["model.safetensors.index.json"] = b'{"weight_map":["bad"]}'
        asset = self.app.store.create(
            "model",
            "invalid index",
            [{"path": path, "size": len(content)} for path, content in files.items()],
        )
        for index, content in enumerate(files.values()):
            self.app.store.chunk(asset["id"], index, 0, content)
        code, reply = self.request("POST", f'/api/uploads/{asset["id"]}/complete', {})
        self.assertEqual(code, 400)
        self.assertIn("shard filenames", reply["error"])


if __name__ == "__main__":
    unittest.main()
