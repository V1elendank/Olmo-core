"""Panel startup and SSH argument checks; no real SSH or cluster access."""
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftlib import status
from ftlib.web import make_server


class PanelTests(unittest.TestCase):
    def test_slow_slurm_collection_does_not_block_upload_api(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            created, collecting, release = threading.Event(), threading.Event(), threading.Event()
            servers = []

            def create(app, port, dashboard):
                server = make_server(app, port, dashboard)
                servers.append(server)
                created.set()
                return server

            def collect():
                collecting.set()
                release.wait(5)
                return {}

            with patch("ftlib.status._root", return_value=root), patch(
                "ftlib.web.make_server", side_effect=create
            ), patch("ftlib.status.collect", side_effect=collect), patch("ftlib.status.write"):
                thread = threading.Thread(target=status.serve, args=(0,), daemon=True)
                thread.start()
                try:
                    self.assertTrue(created.wait(3))
                    self.assertTrue(collecting.wait(3))
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", servers[0].server_port, timeout=2
                    )
                    conn.request("GET", "/api/session")
                    response = conn.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn("token", json.loads(response.read()))
                    conn.close()
                finally:
                    release.set()
                    if servers:
                        servers[0].shutdown()
                    thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

    def run_launcher(self, port="8765", ssh_exit=0):
        """Capture SSH arguments and simulate its exit without opening a connection."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scripts = root / "finetune" / "scripts"
            scripts.mkdir(parents=True)
            launcher = scripts / "launch_from_mac.sh"
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "scripts" / launcher.name, launcher
            )
            fakebin = root / "bin"
            fakebin.mkdir()
            capture = root / "ssh-args"
            (fakebin / "ssh").write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@" > "$SSH_CAPTURE"\nexit "$SSH_EXIT"\n'
            )
            # The test needs no ~/.ssh creation and must not open the user's browser.
            (fakebin / "mkdir").write_text("#!/bin/sh\nexit 0\n")
            for file in fakebin.iterdir():
                file.chmod(0o755)
            result = subprocess.run(
                ["bash", str(launcher), "dgg6b", "panel"],
                cwd=root,
                env=dict(
                    os.environ,
                    PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
                    PORT=port,
                    SSH_CAPTURE=str(capture),
                    SSH_EXIT=str(ssh_exit),
                ),
                text=True,
                capture_output=True,
                timeout=5,
            )
            return result, capture.read_text().splitlines() if capture.exists() else []

    def test_panel_owns_loopback_tunnel_and_propagates_ssh_failure(self):
        result, args = self.run_launcher("8766", 255)
        self.assertEqual(result.returncode, 255)
        self.assertIn("ExitOnForwardFailure=yes", args)
        self.assertIn("ControlMaster=no", args)
        self.assertIn("ControlPersist=no", args)
        self.assertIn("127.0.0.1:8766:127.0.0.1:8766", args)
        self.assertIn("dgg6b@portal.cs.virginia.edu", args)
        self.assertIn("test -f finetune/ftlib/web.py", args[-1])
        self.assertIn("does not sync", result.stdout)

    def test_invalid_port_is_rejected_before_ssh(self):
        for port in ("0", "65536", "abc", "8765; echo bad"):
            with self.subTest(port=port):
                result, args = self.run_launcher(port)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(args, [])

    def test_valid_panel_command_exits_cleanly(self):
        result, args = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("127.0.0.1:8765:127.0.0.1:8765", args)
