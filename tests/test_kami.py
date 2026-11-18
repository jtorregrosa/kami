"""Unit tests for kami (standard library only): `python -m unittest`."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zlib
import unittest
from pathlib import Path
from unittest import mock

from kami import api, catalog, cli, config, fit, guard, jobs, pod, runpod, vast, watchdog

pod_mod = pod   # `pod` is also a common local name in tests


def unpack(program: str) -> str:
    """The source out of pod.pack()'s compressed one-liner."""
    packed = re.search(r"b64decode\('([^']+)'\)", program).group(1)
    return zlib.decompress(base64.b64decode(packed)).decode()


unpack_guard = unpack
from kami.config import ConfigError
from kami.runpod import RunPod, RunPodError

MINIMAL = """
[defaults]
profile = "main"
container = "ghcr.io/me/kami-server:0.1.0"

[profiles.main]
mesh_model = "org/mesh"
gpus = ["GPU A", "GPU B"]
"""


class ProjectTestCase(unittest.TestCase):
    """Runs each test inside a throwaway project directory."""

    toml = MINIMAL

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "kami.toml").write_text(self.toml, encoding="utf-8")
        self.previous = config.project
        config.use_project(str(self.root / "kami.toml"))
        # no real provider keys: nothing (the orphan check, say) may reach a live API
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ("RUNPOD_API_KEY", "VAST_API_KEY", "HF_TOKEN"):
            os.environ.pop(key, None)

    def tearDown(self):
        config.project = self.previous
        self.tmp.cleanup()


class RootCause(unittest.TestCase):
    def test_prefers_specific_error_over_wrapper(self):
        lines = ["INFO starting",
                 "OSError: gated repo facebook/dinov3-vitl16-pretrain-lvd1689m",
                 "RuntimeError: CUDA error: an illegal memory access was encountered"]
        self.assertTrue(pod.root_cause(lines).startswith("OSError"))

    def test_none_without_errors(self):
        self.assertIsNone(pod.root_cause(["all fine"]))


class Profiles(ProjectTestCase):
    toml = MINIMAL + """
[profiles.studio]
extends = "main"
image_model = "org/image"
image_defaults = { steps = 8, guidance = 0.0 }
offload = true
"""

    def test_extends_inherits_and_overrides(self):
        cfg = config.load_config()
        studio = cfg.profiles["studio"]
        self.assertEqual(studio.mesh_model, "org/mesh")
        self.assertEqual(studio.image_model, "org/image")
        self.assertEqual(studio.gpus, ["GPU A", "GPU B"])
        self.assertEqual(studio.image_defaults, {"steps": 8, "guidance": 0.0})
        self.assertTrue(studio.offload)
        self.assertEqual(studio.model, "org/mesh + org/image")

    def test_repos_include_the_mesh_dependencies_once(self):
        cfg = config.load_config()
        repos = cfg.profiles["studio"].repos
        self.assertEqual(repos[:2], ["org/mesh", "org/image"])
        for dep in config.MESH_DEPENDENCIES:
            self.assertEqual(repos.count(dep), 1)
        image_only = config.Profile("i", ["G"], image_model="org/image")
        self.assertEqual(image_only.repos, ["org/image"])

    def test_guard_defaults(self):
        cfg = config.load_config()
        self.assertEqual((cfg.max_hours, cfg.idle_minutes), (4, 30))

    def test_extends_loop(self):
        raw = {"a": {"extends": "b"}, "b": {"extends": "a"}}
        with self.assertRaises(ConfigError):
            config.resolve_profile("a", raw)

    def test_unknown_default_profile(self):
        (self.root / "kami.toml").write_text(MINIMAL.replace('profile = "main"', 'profile = "nope"'),
                                             encoding="utf-8")
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_a_profile_needs_a_model(self):
        (self.root / "kami.toml").write_text(MINIMAL + '\n[profiles.x]\ngpus = ["G"]\n',
                                             encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "needs mesh_model or image_model"):
            config.load_config()

    def test_bad_min_cuda_rejected(self):
        (self.root / "kami.toml").write_text(MINIMAL + 'min_cuda = "12"\n', encoding="utf-8")
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_unknown_job_parameter(self):
        (self.root / "kami.toml").write_text(MINIMAL + 'image_defaults = { stepz = 4 }\n',
                                             encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "unknown parameter 'stepz'"):
            config.load_config()


class FindConfig(unittest.TestCase):
    def test_searches_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "kami.toml").write_text(MINIMAL, encoding="utf-8")
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"KAMI_CONFIG": ""}):
                self.assertEqual(config.find_config(start=nested).resolve(),
                                 (root / "kami.toml").resolve())

    def test_explicit_and_env_win(self):
        self.assertEqual(config.find_config("x.toml"), Path("x.toml"))
        with mock.patch.dict(os.environ, {"KAMI_CONFIG": "y.toml"}):
            self.assertEqual(config.find_config(), Path("y.toml"))


class PodBody(ProjectTestCase):
    def body(self, profile=None, volume=None, **kw):
        cfg = config.load_config()
        return pod.pod_body(cfg, profile or cfg.profile(None), "GPU A", volume, "secret", **kw)

    def test_key_in_env_not_command_line(self):
        body = self.body()
        self.assertEqual(body["env"]["KAMI_API_KEY"], "secret")
        self.assertNotIn("secret", " ".join(body["entrypoint"] + body["cmd"]))

    def test_server_runs_under_the_guard(self):
        body = self.body()
        self.assertEqual(body["entrypoint"][:2], ["python3", "-c"])
        self.assertIn("def main", unpack(body["entrypoint"][2]))
        self.assertIn("[kami-guard]", unpack(body["entrypoint"][2]))
        self.assertEqual(body["entrypoint"][3:], ["python3", "-c", pod.SERVER_BOOT])
        server = unpack(body["env"]["KAMI_SERVER_SRC"])
        self.assertIn("Trellis2ImageTo3DPipeline", server)
        self.assertEqual(body["image"], "ghcr.io/me/kami-server:0.1.0")

    def test_the_models_reach_the_server(self):
        cfg = config.load_config()
        p = config.Profile("s", ["GPU A"], mesh_model="org/mesh", image_model="org/image",
                           image_defaults={"steps": 4}, offload=True, container="my/image:1")
        env = pod.pod_body(cfg, p, "GPU A", None, "k")["env"]
        self.assertEqual((env["KAMI_MESH_MODEL"], env["KAMI_IMAGE_MODEL"]), ("org/mesh", "org/image"))
        self.assertEqual(json.loads(env["KAMI_IMAGE_DEFAULTS"]), {"steps": 4})
        self.assertNotIn("KAMI_MESH_DEFAULTS", env)
        self.assertEqual(env["KAMI_OFFLOAD"], "1")
        self.assertEqual(pod.pod_body(cfg, p, "GPU A", None, "k")["image"], "my/image:1")

    def test_guard_limits_and_watch(self):
        body = self.body()
        self.assertEqual((body["env"]["KAMI_DEADLINE"], body["env"]["KAMI_IDLE"]), ("0", "0"))
        self.assertEqual(body["env"]["KAMI_WATCH"], pod.EPHEMERAL_CACHE)
        self.assertIn("utility", body["env"]["NVIDIA_DRIVER_CAPABILITIES"])
        body = self.body(deadline=1_900_000_000, idle_seconds=3600)
        self.assertEqual(body["env"]["KAMI_DEADLINE"], "1900000000")
        self.assertEqual(body["env"]["KAMI_IDLE"], "3600")

    def test_cuda_12_4_hosts_unless_set(self):
        cfg = config.load_config()
        p = cfg.profile(None)
        self.assertEqual(pod.pod_body(cfg, p, "GPU A", None, "k")["gpu"]["minCudaVersion"], "12.4")
        p.min_cuda = "12.8"
        self.assertEqual(pod.pod_body(cfg, p, "GPU A", None, "k")["gpu"]["minCudaVersion"], "12.8")

    def test_volume_keeps_the_weights(self):
        env = self.body(volume="vol-1")["env"]
        self.assertTrue(env["HF_HOME"].startswith(pod.VOLUME_MOUNT))
        self.assertEqual(env["KAMI_WATCH"], env["HF_HOME"])
        self.assertNotIn("TORCH_HOME", self.body()["env"])

    def test_hf_token_is_passed_when_set(self):
        self.assertNotIn("HF_TOKEN", self.body()["env"])
        os.environ["HF_TOKEN"] = "hf_x"
        self.assertEqual(self.body()["env"]["HF_TOKEN"], "hf_x")


class BootReports(unittest.TestCase):
    def report(self, total=10e9, gpu_bar=True):
        progress = mock.Mock(text="Starting the server")
        progress.phase.side_effect = lambda text: setattr(progress, "text", text)
        return cli.BootReport(progress, total, gpu_bar), progress

    def test_download_switches_phase_and_draws_a_bar(self):
        report, progress = self.report()
        line = "[kami-guard] progress t=100 downloaded=4000000000 rate=200000000 vram=500/24000"
        self.assertTrue(report.line(line))
        self.assertEqual(progress.text, "Downloading the model weights")
        fraction, detail = progress.bar.call_args[0]
        self.assertAlmostEqual(fraction, 0.4)
        self.assertEqual(detail, "4.0/10.0 GB · 200 MB/s · 30s left")

    def test_stale_lines_are_ignored(self):
        report, progress = self.report()
        report.line("[kami-guard] progress t=200 downloaded=5000000000 rate=1")
        report.line("[kami-guard] progress t=100 downloaded=1000000000 rate=1")
        self.assertEqual(report.got, 5_000_000_000)

    def test_gpu_bar_once_loading_and_a_download_summary(self):
        report, progress = self.report()
        report.line("[kami-guard] progress t=100 downloaded=5000000000 rate=1 vram=0/24000")
        report.line("[kami-guard] progress t=150 downloaded=10000000000 rate=1 vram=0/24000")
        progress.phase(cli.LOAD_PHASE)
        with mock.patch.object(cli.ui, "ok") as ok:
            report.line("[kami-guard] progress t=160 downloaded=10000000000 rate=0 vram=4768/24000")
            report.summary()
        ok.assert_called_once()
        self.assertIn("10.0 GB in 50s", ok.call_args[0][0])
        fraction, detail = progress.bar.call_args[0]
        self.assertAlmostEqual(fraction, 0.5, places=2)
        self.assertEqual(detail, "4.7/23 GiB on the GPU")

    def test_more_than_expected_grows_the_total(self):
        report, progress = self.report(total=4e9)
        report.line("[kami-guard] progress t=1 downloaded=3000000000 rate=1")
        report.line("[kami-guard] progress t=4 downloaded=5000000000 rate=1")
        self.assertEqual(progress.bar.call_args[0][0], 1.0)
        self.assertEqual(report.total, 5e9)

    def test_no_gpu_fraction_when_the_weights_shrink_on_the_gpu(self):
        report, progress = self.report(gpu_bar=False)
        progress.phase(cli.LOAD_PHASE)
        report.line("[kami-guard] progress t=1 downloaded=0 rate=0 vram=8000/80000")
        self.assertIsNone(progress.bar.call_args[0][0])

    def test_checks_are_shown_once_and_kept_for_failure_logs(self):
        report, _ = self.report()
        line = "[kami-guard] check warn Hugging Face unreachable from the pod: timed out"
        with mock.patch.object(cli.ui, "warn") as warn:
            self.assertFalse(report.line(line))
            self.assertFalse(report.line(line))
        warn.assert_called_once_with("Hugging Face unreachable from the pod: timed out")


class ProgressBar(unittest.TestCase):
    def test_a_phase_change_clears_the_bar(self):
        progress = cli.ui.Progress("a")
        progress.bar(1.5, "x")
        self.assertEqual((progress.fraction, progress.detail), (1.0, "x"))
        progress.phase("b")
        self.assertEqual((progress.fraction, progress.detail), (None, ""))


class Retries(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(api.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_get_retries_transient_errors(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=[RunPodError(0, "down"),
                                                          RunPodError(502, "bad gateway"),
                                                          {"id": "p"}]) as send:
            self.assertEqual(rp.get_pod("p"), {"id": "p"})
        self.assertEqual(send.call_count, 3)

    def test_post_is_never_retried(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=RunPodError(503, "busy")) as send:
            with self.assertRaises(RunPodError):
                rp.create_pod({})
        self.assertEqual(send.call_count, 1)

    def test_client_errors_are_not_retried(self):
        rp = RunPod("k")
        with mock.patch.object(rp, "_send", side_effect=RunPodError(404, "nope")) as send:
            with self.assertRaises(RunPodError):
                rp.get_pod("p")
        self.assertEqual(send.call_count, 1)


class FakeRunPod:
    name, label = "runpod", "RunPod"

    def __init__(self, pod=None, error=None):
        self.pod, self.error, self.deleted = pod, error, []

    def get_pod(self, pod_id, timeout=None):
        if self.error:
            raise self.error
        return self.pod

    def delete_pod(self, pod_id):
        self.deleted.append(pod_id)

    def account(self):
        return {"balance": 10.0, "spend_per_hour": 0.0}


class WatchdogCheck(ProjectTestCase):
    def setUp(self):
        super().setUp()
        self.state = {"pod_id": "p1", "api_key": "k", "volume_id": "v"}
        config.save_state(self.state)
        patcher = mock.patch.object(watchdog.mascot, "show")
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, rp, limit=3600):
        with mock.patch("sys.stdout"):
            return watchdog._wd_check(rp, "p1", limit, 0)

    def test_under_limit_returns_pod(self):
        rp = FakeRunPod({"status": "RUNNING", "runtime": {"uptime": 10}})
        self.assertIsNotNone(self.check(rp))
        self.assertEqual(rp.deleted, [])

    def test_over_limit_deletes(self):
        rp = FakeRunPod({"status": "RUNNING", "runtime": {"uptime": 4000}, "cost": 1.0})
        self.assertIsNone(self.check(rp))
        self.assertEqual(rp.deleted, ["p1"])
        self.assertNotIn("pod_id", config.load_state())
        self.assertEqual(config.load_state()["volume_id"], "v")   # the volume is kept

    def test_dead_pod_is_deleted(self):
        rp = FakeRunPod({"status": "EXITED"})
        self.assertIsNone(self.check(rp))
        self.assertEqual(rp.deleted, ["p1"])

    def test_gone_pod_clears_state(self):
        self.assertIsNone(self.check(FakeRunPod(error=RunPodError(404, "gone"))))
        self.assertNotIn("pod_id", config.load_state())

    def test_transient_error_propagates_for_retry(self):
        with self.assertRaises(RunPodError) as ctx:
            self.check(FakeRunPod(error=RunPodError(0, "offline")))
        self.assertTrue(ctx.exception.transient)


class StaleState(ProjectTestCase):
    """A long-running `up` / `watchdog` must not unregister a pod created after `kami down`."""

    def setUp(self):
        super().setUp()
        # p1 was deleted with `kami down`; a new `up` then registered p2
        config.save_state({"pod_id": "p2", "api_key": "k2", "volume_id": "v",
                           "history": [{"pod_id": "p1"}, {"pod_id": "p2"}]})

    def test_terminate_own_leaves_newer_pod_registered(self):
        rp = FakeRunPod()
        pod.terminate_own(rp, "p1")
        self.assertEqual(rp.deleted, ["p1"])
        state = config.load_state()
        self.assertEqual(state["pod_id"], "p2")
        self.assertIn("deleted_at", state["history"][0])
        self.assertNotIn("deleted_at", state["history"][1])

    def test_terminate_own_clears_its_own_pod(self):
        rp = FakeRunPod()
        pod.terminate_own(rp, "p2")
        self.assertEqual(rp.deleted, ["p2"])
        self.assertNotIn("pod_id", config.load_state())

    def test_watchdog_of_old_pod_keeps_new_one(self):
        with mock.patch("sys.stdout"):
            watchdog._wd_check(FakeRunPod(error=RunPodError(404, "gone")), "p1", 3600, 0)
        self.assertEqual(config.load_state()["pod_id"], "p2")

    def test_abort_up_of_old_pod_keeps_new_one(self):
        rp = FakeRunPod()
        with mock.patch.object(cli.mascot, "show"), mock.patch("sys.stdout"),                 self.assertRaises(ConfigError):
            cli.abort_up(rp, "p1", ["boom"], keep=False, reason="test")
        self.assertEqual(rp.deleted, ["p1"])
        self.assertEqual(config.load_state()["pod_id"], "p2")


class GuardActivity(unittest.TestCase):
    METRICS = """# HELP kami:jobs_running Jobs on the GPU
kami:jobs_running 1
kami:jobs_queued 2
kami:jobs_finished_total 5
kami:jobs_failed_total 3
"""

    def test_parses_busy_and_finished(self):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = self.METRICS.encode()
        with mock.patch.object(guard.urllib.request, "urlopen", return_value=resp):
            self.assertEqual(guard.activity("8000"), (3.0, 8.0))

    def test_sends_the_pod_key(self):
        # kami-server only leaves /health public: /metrics needs the API key
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = self.METRICS.encode()
        with mock.patch.dict(os.environ, {"KAMI_API_KEY": "k1"}), \
             mock.patch.object(guard.urllib.request, "urlopen", return_value=resp) as urlopen:
            guard.activity("8000")
        self.assertEqual(urlopen.call_args[0][0].get_header("Authorization"), "Bearer k1")

    def test_refused_metrics_are_reported(self):
        err = guard.urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with mock.patch.object(guard, "_refused", False), \
             mock.patch.object(guard.urllib.request, "urlopen", side_effect=err), \
             mock.patch.object(guard, "log") as log:
            self.assertIsNone(guard.activity("8000"))
            self.assertIsNone(guard.activity("8000"))
        log.assert_called_once()      # said once, not every 30 s

    def test_loading_503_is_not_a_refusal(self):
        err = guard.urllib.error.HTTPError("u", 503, "Loading model", {}, None)
        with mock.patch.object(guard, "_refused", False), \
             mock.patch.object(guard.urllib.request, "urlopen", side_effect=err), \
             mock.patch.object(guard, "log") as log:
            self.assertIsNone(guard.activity("8000"))
        log.assert_not_called()

    def test_none_while_server_is_down(self):
        with mock.patch.object(guard.urllib.request, "urlopen", side_effect=OSError("refused")):
            self.assertIsNone(guard.activity("8000"))

    def test_idle_triggers_deletion(self):
        clock = iter(range(0, 10_000, 600))     # each check is 10 minutes later
        with mock.patch.object(guard, "activity", return_value=(0.0, 4.0)), \
             mock.patch.object(guard.time, "monotonic", side_effect=lambda: next(clock)), \
             mock.patch.object(guard.time, "sleep"), \
             mock.patch.object(guard, "log"), \
             mock.patch.object(guard, "delete_self", return_value=True) as delete:
            guard.watch(deadline=0, idle=1800, port="8000")
        delete.assert_called_once()

    def test_deadline_triggers_deletion(self):
        with mock.patch.object(guard.time, "time", return_value=2_000), \
             mock.patch.object(guard.time, "sleep"), \
             mock.patch.object(guard, "log"), \
             mock.patch.object(guard, "delete_self", return_value=True) as delete:
            guard.watch(deadline=1_000, idle=0, port="8000")
        delete.assert_called_once()


class StatusExitCodes(ProjectTestCase):
    def run_status(self):
        args = mock.Mock(json=True)
        with mock.patch("builtins.print"):
            return cli.cmd_status(args, config.load_config())

    def test_no_pod(self):
        self.assertEqual(self.run_status(), cli.STATUS_NO_POD)

    def test_waiting_for_stock_is_not_ready(self):
        now = cli.time.time()
        config.save_state({"waiting": {"profile": "main", "clouds": ["COMMUNITY"],
                                       "since": now - 300, "heartbeat": now - 5}})
        with mock.patch.object(cli, "client", side_effect=ConfigError("no key")):
            self.assertEqual(self.run_status(), cli.STATUS_NOT_READY)

    def test_stale_wait_is_ignored(self):
        old = cli.time.time() - 3600       # that `kami up` died an hour ago
        config.save_state({"waiting": {"profile": "main", "clouds": ["COMMUNITY"],
                                       "since": old, "heartbeat": old}})
        with mock.patch.object(cli, "client", side_effect=ConfigError("no key")):
            self.assertEqual(self.run_status(), cli.STATUS_NO_POD)

    def test_ready_pod(self):
        config.save_state({"pod_id": "p1"})
        rp = FakeRunPod({"id": "p1", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 1.0})
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(cli, "healthy", return_value=True), \
             mock.patch.object(cli, "server_metrics", return_value={"kami:jobs_finished_total": 0}):
            self.assertEqual(self.run_status(), cli.STATUS_READY)


class Costs(ProjectTestCase):
    def test_estimate_counts_from_creation_not_uptime(self):
        # 10 min since creation, but the container has only been up 2 min (image pull)
        pod = {"runtime": {"uptime": 120}, "cost": 0.60}
        with mock.patch.object(pod_mod.time, "time", return_value=1_000_600):
            text = pod_mod.session_cost(pod, since=1_000_000)
        self.assertIn("$0.10", text)
        self.assertIn("estimate", text)

    def test_real_cost_is_the_balance_drop(self):
        rp = mock.Mock(account=mock.Mock(return_value={"balance": 14.66, "spend_per_hour": 0}))
        self.assertAlmostEqual(pod_mod.real_session_cost(rp, {"balance_at_up": 14.82}), 0.16)
        self.assertIsNone(pod_mod.real_session_cost(rp, {}))      # no balance recorded
        # a top-up meanwhile: unknown, not $0.00
        self.assertIsNone(pod_mod.real_session_cost(rp, {"balance_at_up": 10.0}))

    def down_text(self, session, real, estimate):
        with mock.patch.dict(cli.os.environ, {"NO_COLOR": "1"}):
            return "\n".join(cli.down_cost_lines(session, real, estimate, "RunPod"))

    def test_down_shows_the_real_cost_once_debited(self):
        text = self.down_text("27m · ≈ $0.79 (estimate)", 0.78, 0.79)
        self.assertIn("Real cost: $0.78", text)

    def test_down_never_reports_an_undebited_session_as_free(self):
        # RunPod debits in batches: 6 minutes in, nothing has left the balance yet
        text = self.down_text("6m · ≈ $0.16 (estimate)", 0.0, 0.16)
        self.assertNotIn("$0.00", text)
        self.assertIn("≈ $0.16", text)
        self.assertIn("has not debited it yet", text)
        text = self.down_text(None, 0.14, 0.31)    # pod already gone: the estimate from history
        self.assertIn("Session: ≈ $0.31", text)
        self.assertIn("has debited $0.14 of it so far", text)
        self.assertNotIn("Real cost", text)
        self.assertNotIn("$0.00", self.down_text(None, 0.0, None))

    def test_history_survives_the_pod_and_marks_deletion(self):
        state = {}
        pod_mod.remember_pod(state, {"id": "p1", "gpu": {"id": "G"}, "cost": 0.5}, "basic-general")
        state.update(pod_id="p1", api_key="k", created_at=1, balance_at_up=9)
        config.save_state(state)
        pod_mod.terminate(FakeRunPod(), state)
        saved = config.load_state()
        self.assertNotIn("pod_id", saved)
        self.assertNotIn("balance_at_up", saved)
        self.assertEqual(saved["history"][0]["profile"], "basic-general")
        self.assertIn("deleted_at", saved["history"][0])

    def test_estimate_from_history_rate(self):
        state = {"pod_id": "p1", "created_at": 1_000_000,
                 "history": [{"pod_id": "p1", "cost_per_hour": 2.49}]}
        with mock.patch.object(cli.time, "time", return_value=1_000_000 + 449):
            self.assertAlmostEqual(cli.estimated_cost(state), 0.31, places=2)
        self.assertIsNone(cli.estimated_cost({"pod_id": "p1"}))

    def test_history_is_capped(self):
        state = {}
        for i in range(pod_mod.HISTORY_MAX + 5):
            pod_mod.remember_pod(state, {"id": f"p{i}"}, "x")
        self.assertEqual(len(state["history"]), pod_mod.HISTORY_MAX)
        self.assertEqual(state["history"][-1]["pod_id"], f"p{pod_mod.HISTORY_MAX + 4}")

    def test_billing_rows_are_summed_per_pod(self):
        rows = [{"podId": "a", "amount": 1.1, "timeBilledMs": 3_600_000, "time": "2026-10-21 05:00:00"},
                {"podId": "a", "amount": 0.1, "timeBilledMs": 300_000, "time": "2026-10-21 04:00:00"},
                {"podId": "b", "amount": 0.2, "timeBilledMs": 60_000, "time": "2026-10-21 06:00:00"}]
        pods = cli.billed_by_pod(rows)
        self.assertAlmostEqual(pods["a"]["amount"], 1.2)
        self.assertAlmostEqual(pods["a"]["minutes"], 65)
        self.assertEqual(pods["a"]["first"], "2026-10-21 04:00:00")


class UpCosts(unittest.TestCase):
    """The $/h `up`, `status` and the estimates use is what the pod really bills."""

    def test_runpod_cost_includes_the_container_disk(self):
        # seen live: a $1.69/h GPU with 200 GB of disk -> the account spends $1.718/h
        pod = runpod.with_disk_cost({"cost": 1.69, "disk": 200})
        self.assertAlmostEqual(pod["cost"], 1.7178, places=4)
        self.assertEqual(pod["gpu_cost"], 1.69)
        self.assertEqual(runpod.with_disk_cost(pod)["cost"], pod["cost"])   # not added twice

    def test_hourly_text_breaks_the_price_down(self):
        with mock.patch.dict(cli.os.environ, {"NO_COLOR": "1"}):
            text = cli.hourly_text({"cost": 1.7178, "gpu_cost": 1.69})
        self.assertIn("$1.72/h", text)
        self.assertIn("GPU $1.69 + disk $0.03", text)
        self.assertEqual(cli.hourly_text({}), "$?/h")

    def test_download_is_priced_on_vast_only(self):
        profile = mock.Mock()
        self.assertIsNone(cli.download_text({"cost": 1.0}, profile))     # RunPod: free
        with mock.patch.object(cli.fit, "download_gb", return_value=120.0):
            text = cli.download_text({"download_cost_per_gb": 0.01}, profile)
        self.assertIn("≈ $1.20 once", text)
        with mock.patch.object(cli.fit, "download_gb", side_effect=LookupError("x")):
            self.assertIn("$10.00/TB", cli.download_text({"download_cost_per_gb": 0.01}, profile))


class Snapshot(unittest.TestCase):
    def wait_done(self, snap):
        for _ in range(200):
            if snap.done:
                return
            cli.time.sleep(0.01)
        self.fail("snapshot never finished")

    def test_reads_lines_and_closes_the_stream(self):
        resp = mock.Mock()
        rp = mock.Mock(open_logs=mock.Mock(return_value=resp),
                       stream_logs=mock.Mock(return_value=iter([{"line": "a"}, {"line": "b"}])))
        snap = pod_mod.LogSnapshot(rp, "p1", wait=5)
        self.wait_done(snap)
        self.assertEqual(snap.lines, ["a", "b"])
        resp.close.assert_called()

    def test_errors_are_swallowed(self):
        rp = mock.Mock(open_logs=mock.Mock(side_effect=RunPodError(0, "offline")))
        snap = pod_mod.LogSnapshot(rp, "p1")
        self.wait_done(snap)
        self.assertEqual(snap.lines, [])


class WaitForStock(unittest.TestCase):
    NO_STOCK = RunPodError(500, "There are no longer any instances available with the requested specifications.")
    BAD = RunPodError(400, "invalid image")

    def run_create(self, side_effect, clouds=("COMMUNITY",), max_wait=0):
        rp = mock.Mock(create_pod=mock.Mock(side_effect=side_effect))
        sleep = mock.Mock()
        cands = [fit.Fit("GPU A", None, None), fit.Fit("GPU B", None, None)]
        with mock.patch("builtins.print"), mock.patch("sys.stdout"):
            result = cli.create_with_stock(rp, mock.Mock(datacenter=""), cands, list(clouds),
                                           lambda gpu, cloud: {"gpu": gpu, "cloud": cloud},
                                           max_wait=max_wait, sleep=sleep)
        return result, rp, sleep

    def test_waits_until_stock_appears(self):
        # round 1: both GPUs out of stock -> wait; round 2: GPU A has stock
        (pod, cand, cloud), rp, sleep = self.run_create([self.NO_STOCK, self.NO_STOCK, {"id": "p"}])
        self.assertEqual((pod["id"], cand.gpu, cloud), ("p", "GPU A", "COMMUNITY"))
        self.assertEqual(sleep.call_count, cli.STOCK_POLL)    # exactly one wait round

    def test_secure_is_tried_after_community_without_waiting(self):
        (pod, cand, cloud), rp, sleep = self.run_create(
            [self.NO_STOCK, self.NO_STOCK, {"id": "p"}], clouds=("COMMUNITY", "SECURE"))
        self.assertEqual(cloud, "SECURE")
        self.assertEqual(rp.create_pod.call_args[0][0], {"gpu": "GPU A", "cloud": "SECURE"})
        sleep.assert_not_called()

    def test_refused_request_does_not_wait(self):
        with self.assertRaises(ConfigError):
            self.run_create([self.BAD, self.BAD])

    def test_max_wait_gives_up(self):
        with mock.patch.object(cli.time, "monotonic", side_effect=[0, 0, 10_000, 10_000]), \
             self.assertRaises(ConfigError):
            self.run_create([self.NO_STOCK] * 10, max_wait=60)


class MemoryFit(unittest.TestCase):
    CATALOG = [{"id": "96GB", "memory": 96}, {"id": "24GB", "memory": 24}]

    def profile(self, need=40, count=1):
        return config.Profile("x", ["24GB", "96GB", "unknown"], mesh_model="m",
                              min_vram_gb=need, gpu_count=count)

    def test_fits_per_gpu(self):
        results = fit.fits(self.profile(), self.CATALOG)
        self.assertEqual([f.gpu for f in results], ["24GB", "96GB", "unknown"])
        self.assertEqual([f.fits for f in results], [False, True, True])   # unknown: not judged

    def test_no_requirement_fits_everything(self):
        self.assertTrue(all(f.fits for f in fit.fits(self.profile(need=0), self.CATALOG)))

    def test_several_gpus_pool_their_memory(self):
        self.assertTrue(fit.fits(self.profile(count=2), self.CATALOG)[0].fits)

    def test_up_stops_before_creating_a_pod(self):
        rp = mock.Mock(list_gpus=mock.Mock(return_value=self.CATALOG))
        p = config.Profile("x", ["24GB"], mesh_model="m", min_vram_gb=40)
        with mock.patch("builtins.print"), self.assertRaises(ConfigError):
            cli.memory_check(rp, mock.Mock(cloud="SECURE"), p)
        rp.create_pod.assert_not_called()

    def test_skips_the_small_gpus(self):
        rp = mock.Mock(list_gpus=mock.Mock(return_value=self.CATALOG))
        with mock.patch.object(cli.ui, "warn") as warn:
            usable = cli.memory_check(rp, mock.Mock(cloud="SECURE"), self.profile())
        self.assertEqual([f.gpu for f in usable], ["96GB", "unknown"])
        warn.assert_called_once()


class GpuTable(unittest.TestCase):
    def test_every_spelling_has_one_key(self):
        for name in ("RTX 4090", "NVIDIA GeForce RTX 4090", "rtx 4090"):
            self.assertEqual(catalog.key(name), "RTX 4090")
        self.assertEqual(catalog.key("RTX PRO 6000 S"), "RTX PRO 6000 Server")
        self.assertEqual(catalog.key("Some new GPU"), "Some new GPU")

    def test_runpod_gets_its_id_or_the_name_untouched(self):
        self.assertEqual(catalog.provider_gpu("L40S", "runpod"), "NVIDIA L40S")
        self.assertEqual(catalog.provider_gpu("NVIDIA L40S", "runpod"), "NVIDIA L40S")
        self.assertEqual(catalog.provider_gpu("NVIDIA B300", "runpod"), "NVIDIA B300")

    def test_vast_never_gets_a_guessed_name(self):
        self.assertEqual(catalog.provider_gpu("NVIDIA GeForce RTX 4090", "vast"), "RTX 4090")
        with self.assertRaises(ConfigError):
            catalog.provider_gpu("NVIDIA B300", "vast")

    def test_repo_profiles_are_in_the_table(self):
        # every GPU of the shipped kami.toml can be rented on both providers
        cfg = config.load_config(config.SOURCE_ROOT / "kami.toml")
        for p in cfg.profiles.values():
            for gpu in p.gpus:
                row = catalog.find(gpu)
                self.assertIsNotNone(row, gpu)
                self.assertTrue(row.runpod and row.vast and row.vast_max_dph, gpu)


class ProviderConfig(ProjectTestCase):
    def write(self, extra):
        (self.root / "kami.toml").write_text(MINIMAL.replace('[defaults]\n', '[defaults]\n' + extra),
                                              encoding="utf-8")

    def test_default_is_runpod(self):
        self.assertEqual(config.load_config().provider, "runpod")

    def test_vast(self):
        self.write('provider = "vast"\n')
        text = (self.root / "kami.toml").read_text(encoding="utf-8")
        (self.root / "kami.toml").write_text(text.replace('"GPU A", "GPU B"', '"RTX 4090", "L40S"'),
                                              encoding="utf-8")
        self.assertEqual(config.load_config().provider, "vast")

    def test_unknown_provider(self):
        self.write('provider = "lambda"\n')
        with self.assertRaises(ConfigError):
            config.load_config()

    def test_vast_has_no_volumes(self):
        self.write('provider = "vast"\nstorage = "volume"\n')
        with self.assertRaises(ConfigError):
            config.load_config()


class VastBody(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["NVIDIA GeForce RTX 4090", "L40S"]')

    def body(self, gpu="NVIDIA GeForce RTX 4090", cloud="COMMUNITY", **kw):
        cfg = config.load_config()
        return pod.vast_body(cfg, cfg.profile(None), gpu, "secret", cloud=cloud, **kw)

    def test_offer_filters(self):
        offer = self.body()["offer"]
        self.assertEqual(offer["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual(offer["num_gpus"], {"eq": 1})
        self.assertEqual(offer["verified"], {"eq": True})
        self.assertEqual(offer["dph_base"], {"lte": catalog.find("RTX 4090").vast_max_dph})
        self.assertGreater(offer["gpu_ram"]["gte"], 20000)        # MiB: a 24 GB card
        self.assertNotIn("hosting_type", offer)
        self.assertEqual(offer["cuda_max_good"], {"gte": 12.4})     # the server image's CUDA
        self.assertEqual(self.body(cloud="SECURE")["offer"]["hosting_type"], {"eq": 1})

    def test_server_runs_under_the_guard_with_the_port_published(self):
        inst = self.body()["instance"]
        self.assertEqual(inst["runtype"], "args")
        self.assertEqual(inst["entrypoint"], "python3")
        self.assertEqual(inst["args"][0], "-c")
        self.assertIn("def main", unpack_guard(inst["args"][1]))
        self.assertEqual(inst["args"][2:], ["python3", "-c", pod.SERVER_BOOT])
        self.assertEqual(inst["env"]["-p 8000:8000"], "1")
        self.assertEqual(inst["env"]["KAMI_PROVIDER"], "vast")
        self.assertEqual(inst["env"]["KAMI_TUNNEL"], "1")
        self.assertEqual(inst["env"]["KAMI_API_KEY"], "secret")
        self.assertNotIn("secret", inst["args"])
        self.assertEqual(inst["label"], "kami-main")
        self.assertTrue(inst["cancel_unavail"])

    def test_limits_reach_the_guard(self):
        env = self.body(deadline=1_900_000_000, idle_seconds=600)["instance"]["env"]
        self.assertEqual((env["KAMI_DEADLINE"], env["KAMI_IDLE"]), ("1900000000", "600"))

    def test_multi_gpu_caps_the_whole_offer(self):
        cfg = config.load_config()
        p = config.Profile("tp", ["L40S"], mesh_model="org/m", gpu_count=2)
        offer = pod.vast_body(cfg, p, "L40S", "k", cloud="COMMUNITY")["offer"]
        self.assertEqual(offer["num_gpus"], {"eq": 2})
        self.assertEqual(offer["dph_base"], {"lte": catalog.find("L40S").vast_max_dph * 2})

    def test_unknown_gpu_is_refused(self):
        with self.assertRaises(ConfigError):
            self.body(gpu="GPU nobody knows")

    def test_runpod_body_translates_short_names(self):
        cfg = config.load_config()
        body = pod.pod_body(cfg, cfg.profile(None), "L40S", None, "k")
        self.assertEqual(body["gpu"]["id"], "NVIDIA L40S")
        self.assertNotIn("KAMI_PROVIDER", body["env"])


class VastClient(unittest.TestCase):
    INSTANCE = {"id": 42, "actual_status": "running", "dph_total": 0.5, "gpu_name": "RTX 4090",
                "public_ipaddr": "1.2.3.4\n", "start_date": 1_000_000.0,
                "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "33526"}]}}
    OFFER = {"id": 7, "gpu_name": "RTX 4090", "dph_base": 0.40, "storage_cost": 0.36,
             "geolocation": "Norway, NO", "inet_down_cost": 0.01}

    def test_instance_in_runpod_shape(self):
        with mock.patch.object(vast.time, "time", return_value=1_000_060.0):
            p = vast.normalize(self.INSTANCE)
        self.assertEqual((p["id"], p["status"], p["cost"]), ("42", "RUNNING", 0.5))
        self.assertEqual(p["gpu"]["id"], "RTX 4090")
        self.assertEqual(p["runtime"]["uptime"], 60)
        self.assertEqual(pod.endpoint(p, 8000), "http://1.2.3.4:33526")
        self.assertIsNone(pod.endpoint(p, 9000))     # that port is not published

    def test_runpod_pods_keep_the_proxy(self):
        # RunPod's pods carry their own "ports" list: not to be read as vast's mapping
        p = {"id": "abc", "status": "RUNNING", "ports": ["8000/http"]}
        self.assertEqual(pod.endpoint(p, 8000), pod.proxy_url("abc", 8000))

    def test_statuses(self):
        for raw, status in (("loading", "CREATED"), (None, "CREATED"), ("exited", "EXITED")):
            self.assertEqual(vast.normalize({**self.INSTANCE, "actual_status": raw})["status"], status)
        self.assertIn("EXITED", pod.DEAD_STATES)

    def test_missing_instance_is_a_404(self):
        v = vast.Vast("k")
        with mock.patch.object(v, "_send", return_value={"instances": None}):
            with self.assertRaises(vast.VastError) as ctx:
                v.get_pod("42")
        self.assertEqual(ctx.exception.status, 404)

    def test_rents_the_first_offer_left(self):
        v = vast.Vast("k")
        taken = vast.VastError(410, "no_such_ask Instance type 7 is no longer available.")
        offers = [self.OFFER, {**self.OFFER, "id": 8, "dph_base": 0.45}]
        with mock.patch.object(v, "search_offers", return_value=offers), \
             mock.patch.object(v, "_request", side_effect=[taken, {"new_contract": 99}]) as req:
            p = v.create_pod({"offer": {}, "instance": {"disk": 72}})
        self.assertEqual(req.call_args[0][:2], ("PUT", "/asks/8/"))
        self.assertEqual(p["id"], "99")
        self.assertAlmostEqual(p["cost"], 0.45 + 0.36 * 72 / 720, places=3)

    def test_no_offers_is_no_stock(self):
        v = vast.Vast("k")
        with mock.patch.object(v, "search_offers", return_value=[]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertTrue(cli.is_no_stock(ctx.exception))

    def test_all_offers_taken_is_no_stock(self):
        v = vast.Vast("k")
        taken = vast.VastError(410, "no_such_ask")
        with mock.patch.object(v, "search_offers", return_value=[self.OFFER]), \
             mock.patch.object(v, "_request", side_effect=[taken]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertTrue(cli.is_no_stock(ctx.exception))

    def test_refused_rent_is_not_no_stock(self):
        v = vast.Vast("k")
        refused = vast.VastError(400, "Invalid args")
        with mock.patch.object(v, "search_offers", return_value=[self.OFFER]), \
             mock.patch.object(v, "_request", side_effect=[refused]):
            with self.assertRaises(vast.VastError) as ctx:
                v.create_pod({"offer": {}, "instance": {}})
        self.assertFalse(cli.is_no_stock(ctx.exception))

    def test_gpu_catalog_groups_offers(self):
        v = vast.Vast("k")
        offers = [{"gpu_name": "RTX 4090", "gpu_ram": 24564, "dph_total": 0.5, "geolocation": "Norway, NO"},
                  {"gpu_name": "RTX 4090", "gpu_ram": 24564, "dph_total": 0.4, "geolocation": "Texas, US"}]
        with mock.patch.object(v, "search_offers", return_value=offers):
            [g] = v.list_gpus("COMMUNITY")
        self.assertEqual((g["id"], g["memory"], g["price"]["community"]), ("RTX 4090", 24, 0.4))
        self.assertEqual([d["id"] for d in g["dataCenters"]], ["NO", "US"])
        # the memory check reads it like RunPod's catalog
        p = config.Profile("x", ["NVIDIA GeForce RTX 4090"], mesh_model="m", min_vram_gb=24)
        self.assertEqual(fit.fits(p, [g])[0].vram_gb, 24)


class VastLogs(unittest.TestCase):
    def test_new_lines_only(self):
        self.assertEqual(vast.new_lines(["a", "b", "c"], ["b", "c", "d"]), ["d"])
        self.assertEqual(vast.new_lines(["a", "b"], ["a", "b"]), [])
        self.assertEqual(vast.new_lines([], ["a"]), ["a"])
        self.assertEqual(vast.new_lines(["a"], ["x", "y"]), ["x", "y"])   # window moved on

    def test_poll_yields_each_line_once(self):
        v = vast.Vast("k")
        fetches = [["a", "b"], ["a", "b", "c"]]
        poll = vast.LogPoll(v, "1", tail=1)

        def fetch(pod_id, tail):
            if not fetches:
                poll.close()
                return ["a", "b", "c"]
            return fetches.pop(0)

        with mock.patch.object(v, "fetch_logs", side_effect=fetch), \
             mock.patch.object(vast.time, "sleep"):
            self.assertEqual([e["line"] for e in v.stream_logs("1", resp=poll)], ["b", "c"])


class GuardProgress(unittest.TestCase):
    def test_tree_bytes_counts_files_but_not_the_xet_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hub").mkdir()
            (root / "hub" / "a.gguf.downloadInProgress").write_bytes(b"x" * 300)
            (root / "xet").mkdir()
            (root / "xet" / "chunk").write_bytes(b"x" * 1000)
            self.assertEqual(guard.tree_bytes([tmp, str(root / "missing")]), 300)

    def test_logs_the_growth_and_gpu_memory_until_the_server_answers(self):
        with mock.patch.object(guard, "answering", side_effect=[False, False, True]),              mock.patch.object(guard, "tree_bytes", side_effect=[100, 3100, 3100]),              mock.patch.object(guard, "nvidia_smi", return_value=[["900", "24000"]]),              mock.patch.object(guard.time, "monotonic", side_effect=[0, 3, 6]),              mock.patch.object(guard.time, "time", return_value=1_000),              mock.patch.object(guard.time, "sleep"),              mock.patch.object(guard, "log") as log:
            guard.progress(["/d"], "8000")
        # the second sample changed nothing: not logged
        log.assert_called_once_with("progress t=1000 downloaded=3000 rate=1000 vram=900/24000")


class GuardProgram(unittest.TestCase):
    def test_compressed_guard_runs_the_server_and_passes_its_exit_code(self):
        program = pod.guard_program()
        self.assertLess(len(program), len(pod.GUARD_SOURCE.read_bytes()))
        env = {k: v for k, v in os.environ.items() if not k.startswith("KAMI_")}
        done = subprocess.run([sys.executable, "-c", program, sys.executable, "-c", "raise SystemExit(3)"],
                              capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(done.returncode, 3)
        self.assertIn("[kami-guard] no limits set", done.stdout)


class AccessCheck(unittest.TestCase):
    def check(self, files=(("model.safetensors", 5),), head=None, repo="org/repo"):
        opener = mock.Mock()
        if head:
            opener.open.side_effect = head
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(fit, "repo_files", return_value=files) as listing, \
             mock.patch.object(fit.urllib.request, "build_opener", return_value=opener):
            if isinstance(files, Exception):
                listing.side_effect = files
            return fit.repo_problem(repo)

    def http_error(self, code):
        return fit.urllib.error.HTTPError("u", code, "x", {}, None)

    def test_downloadable(self):
        self.assertIsNone(self.check())
        # the CDN redirect is an answer, not an error
        self.assertIsNone(self.check(head=self.http_error(302)))

    def test_gated_repo_says_how_to_get_access(self):
        problem = self.check(head=self.http_error(403))
        self.assertIn("gated", problem)
        self.assertIn("HF_TOKEN", problem)
        self.assertIn("https://huggingface.co/org/repo", problem)

    def test_missing_repo(self):
        self.assertIn("not found", self.check(files=fit.HubError("x", 401)))

    def test_offline_does_not_block(self):
        self.assertIsNone(self.check(files=LookupError("offline")))
        self.assertIsNone(self.check(head=fit.urllib.error.URLError("down")))

    def test_every_repo_of_the_profile_is_checked(self):
        p = config.Profile("x", ["G"], mesh_model="org/mesh", image_model="org/image")
        with mock.patch.object(fit, "repo_problem",
                               side_effect=lambda repo, timeout=10: "gated" if "dinov3" in repo else None) as check:
            self.assertEqual(fit.access_problems(p), ["gated"])
        checked = [c.args[0] for c in check.call_args_list]
        self.assertEqual(checked, p.repos)


class SlowDownload(unittest.TestCase):
    def test_warns_once_after_a_sustained_slow_rate_and_keeps_going(self):
        progress = mock.Mock(text="Starting the server")
        progress.phase.side_effect = lambda text: setattr(progress, "text", text)
        report = cli.BootReport(progress, 20e9, hourly=2.0)
        with mock.patch.object(cli.ui, "warn") as warn, mock.patch.object(cli.ui, "hint"):
            for t in range(0, 200, 10):
                report.line(f"[kami-guard] progress t={t} downloaded={(t + 1) * 10_000_000} rate=10000000")
        warn.assert_called_once()
        self.assertIn("10 MB/s", warn.call_args[0][0])
        self.assertIn("of pod time", warn.call_args[0][0])

    def test_a_fast_burst_resets_the_clock(self):
        progress = mock.Mock(text="Downloading the model weights")
        report = cli.BootReport(progress, 20e9)
        with mock.patch.object(cli.ui, "warn") as warn:
            for t in range(0, 200, 10):
                rate = 10_000_000 if t % 80 else 100_000_000
                report.line(f"[kami-guard] progress t={t} downloaded={(t + 1) * 10_000_000} rate={rate}")
        warn.assert_not_called()


class BootHistory(ProjectTestCase):
    def test_typical_boot_is_the_median_on_the_same_gpu_and_storage(self):
        history = [{"profile": "main", "provider": "runpod", "storage": "ephemeral", "gpu": g,
                    "boot_seconds": s} for g, s in
                   (("A", 300), ("A", 500), ("A", 400), ("B", 900))]
        history.append({"profile": "main", "provider": "runpod", "storage": "volume",
                        "gpu": "A", "boot_seconds": 60})
        state = {"history": history}
        self.assertEqual(pod.typical_boot(state, "main", "runpod", "ephemeral", "A"), (400, 3))
        # an unseen GPU falls back to every boot of the profile
        self.assertEqual(pod.typical_boot(state, "main", "runpod", "ephemeral", "C"), (450, 4))
        self.assertEqual(pod.typical_boot(state, "main", "runpod", "volume"), (60, 1))
        self.assertIsNone(pod.typical_boot(state, "other", "runpod", "ephemeral"))

    def test_record_boot_updates_the_pod_entry(self):
        config.save_state({"history": [{"pod_id": "p1"}, {"pod_id": "p2"}]})
        pod.record_boot("p2", boot_seconds=321)
        self.assertEqual(config.load_state()["history"][1]["boot_seconds"], 321)
        self.assertNotIn("boot_seconds", config.load_state()["history"][0])


class JobClient(unittest.TestCase):
    def answer(self, status, body):
        return status, json.dumps(body).encode()

    def test_wait_follows_the_job_until_done(self):
        states = [self.answer(200, {"id": "j", "status": "running"}),
                  self.answer(200, {"id": "j", "status": "done", "files": ["owl.glb"]})]
        with mock.patch.object(jobs, "_request", side_effect=states), \
             mock.patch.object(jobs.ui, "Progress"):
            job = jobs.wait("https://x", "k", {"id": "j", "status": "queued"}, "Mesh", sleep=lambda s: None)
        self.assertEqual(job["files"], ["owl.glb"])

    def test_a_failed_job_says_why(self):
        with mock.patch.object(jobs.ui, "Progress"), \
             self.assertRaisesRegex(jobs.JobError, "CUDA out of memory"):
            jobs.wait("https://x", "k", {"id": "j", "status": "error",
                                         "error": "OutOfMemoryError: CUDA out of memory"}, "Mesh")

    def test_http_errors_carry_the_servers_message(self):
        with mock.patch.object(jobs, "_request", return_value=self.answer(400, {"error": "a mesh job needs an image"})), \
             self.assertRaisesRegex(jobs.JobError, "a mesh job needs an image"):
            jobs.submit("https://x", "k", {"kind": "mesh"})
        with mock.patch.object(jobs, "_request", return_value=self.answer(401, {})), \
             self.assertRaisesRegex(jobs.JobError, "API key"):
            jobs.info("https://x", "k")

    def test_download_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "owl.glb").write_bytes(b"old")
            with mock.patch.object(jobs, "_request", return_value=(200, b"glTF")):
                [path] = jobs.download("https://x", "k", {"id": "j", "files": ["owl.glb"]}, out)
            self.assertEqual(path.name, "owl-2.glb")
            self.assertEqual((out / "owl.glb").read_bytes(), b"old")

    def test_read_image_checks_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "a.png"
            img.write_bytes(b"\x89PNG")
            self.assertEqual(base64.b64decode(jobs.read_image(img)), b"\x89PNG")
            with self.assertRaisesRegex(jobs.JobError, "not an image"):
                (Path(tmp) / "a.txt").write_text("x")
                jobs.read_image(Path(tmp) / "a.txt")
            with self.assertRaisesRegex(jobs.JobError, "not found"):
                jobs.read_image(Path(tmp) / "missing.png")

    def test_names(self):
        self.assertEqual(jobs.safe_name("A ceramic owl, studio light!"), "a-ceramic-owl-studio-light")
        self.assertEqual(jobs.safe_name("!!!"), "kami")


class Server(unittest.TestCase):
    """server.py's request handling, without torch (the pipelines are mocked)."""

    def setUp(self):
        from kami import server
        self.server = server
        patcher = mock.patch.multiple(server, JOBS={}, QUEUE_ORDER=[], QUEUE=mock.Mock())
        patcher.start()
        self.addCleanup(patcher.stop)
        pipes = mock.patch.object(server, "PIPES", mock.Mock(mesh=object(), image=None, ready=True))
        pipes.start()
        self.addCleanup(pipes.stop)
        log = mock.patch.object(server, "log")
        log.start()
        self.addCleanup(log.stop)

    def test_a_mesh_job_is_queued(self):
        job = self.server.submit({"kind": "mesh", "image": base64.b64encode(b"png").decode(),
                                  "name": "owl/../x"})
        self.assertEqual(job.status, "queued")
        self.assertEqual(job.name, "owlx")                   # no path tricks in file names
        self.assertEqual(job.info()["position"], 1)
        self.server.QUEUE.put.assert_called_once_with(job)

    def test_bad_requests(self):
        for body, error in (({"kind": "video"}, "kind must be"),
                            ({"kind": "mesh"}, "needs an image"),
                            ({"kind": "mesh", "image": "%%%"}, "base64"),
                            ({"kind": "image", "prompt": "x"}, "no image model"),
                            ({"kind": "text-to-mesh", "prompt": "x"}, "no image model")):
            with self.assertRaisesRegex(ValueError, error):
                self.server.submit(body)

    def test_the_guard_reads_the_servers_metrics(self):
        # the names the guard watches are the ones the server publishes
        source = Path(self.server.__file__).read_text(encoding="utf-8")
        for name in (*guard.BUSY, *guard.DONE):
            self.assertIn(name, source)


class ServerRoundTrip(unittest.TestCase):
    """The real HTTP server and worker against kami's job client, over localhost; only
    the GPU work (server.run) is replaced."""

    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from kami import server
        self.server = server
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        def fake_run(job):
            os.makedirs(job.dir, exist_ok=True)
            if job.image_bytes == b"boom":
                raise RuntimeError("CUDA out of memory")
            with open(os.path.join(job.dir, f"{job.name}.glb"), "wb") as f:
                f.write(b"glTF" + job.image_bytes)
            job.files.append(f"{job.name}.glb")

        for patcher in (mock.patch.multiple(server, API_KEY="k", JOBS_DIR=self.tmp.name, JOBS={},
                                            QUEUE_ORDER=[], run=fake_run, log=mock.Mock(),
                                            COUNTS={"finished": 0, "failed": 0, "running": 0}),
                        mock.patch.object(server, "PIPES", mock.Mock(mesh=object(), image=None,
                                                                     ready=True, error=None)),
                        mock.patch.object(server.traceback, "print_exc")):
            patcher.start()
            self.addCleanup(patcher.stop)
        server.QUEUE = __import__("queue").Queue()
        threading.Thread(target=server.worker, daemon=True).start()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def mesh(self, image: bytes, name="owl"):
        job = jobs.submit(self.url, "k", {"kind": "mesh", "name": name,
                                          "image": base64.b64encode(image).decode()})
        with mock.patch.object(jobs, "POLL_EVERY", 0), mock.patch.object(jobs.ui, "Progress"):
            return jobs.wait(self.url, "k", job, "Mesh", sleep=lambda s: None)

    def test_submit_wait_download(self):
        self.assertTrue(pod.healthy(self.url))
        job = self.mesh(b"png")
        out = Path(self.tmp.name) / "out"
        [path] = jobs.download(self.url, "k", job, out)
        self.assertEqual(path.read_bytes(), b"glTFpng")
        usage = pod.server_usage(pod.server_metrics(self.url, "k"))
        self.assertEqual(usage["finished"], 1)

    def test_a_failure_reaches_the_client(self):
        with self.assertRaisesRegex(jobs.JobError, "CUDA out of memory"):
            self.mesh(b"boom")
        self.assertEqual(pod.server_usage(pod.server_metrics(self.url, "k"))["failed"], 1)

    def test_the_key_is_required(self):
        with self.assertRaisesRegex(jobs.JobError, "API key"):
            jobs.info(self.url, "wrong")


class ServerUsage(unittest.TestCase):
    def test_jobs(self):
        usage = pod.server_usage({"kami:jobs_running": 1, "kami:jobs_queued": 2,
                                  "kami:jobs_finished_total": 7, "kami:jobs_failed_total": 1})
        self.assertEqual(usage, {"running": 1, "queued": 2, "finished": 7, "failed": 1})
        self.assertIsNone(pod.server_usage({}))

    def test_metrics_are_summed_over_labels(self):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = (
            b'# HELP x\nkami:jobs_running{a="0"} 1\n'
            b'kami:jobs_finished_total{k="mesh"} 5\nkami:jobs_finished_total{k="image"} 7\n')
        with mock.patch.object(pod.urllib.request, "urlopen", return_value=resp):
            metrics = pod.server_metrics("https://x", "k")
        self.assertEqual(metrics, {"kami:jobs_running": 1, "kami:jobs_finished_total": 12})

    def test_status_lines(self):
        self.assertEqual(cli.usage_lines(None), [])
        idle = cli.usage_lines({"running": 0, "queued": 0, "finished": 0, "failed": 0})
        self.assertIn("none yet", idle[0])
        [line] = cli.usage_lines({"running": 1, "queued": 0, "finished": 7, "failed": 1})
        self.assertIn("7 done", line)
        self.assertIn("1 failed", line)
        self.assertIn("1 running", line)


class LogsFilter(ProjectTestCase):
    def test_progress_samples_hidden_unless_verbose(self):
        config.save_state({"pod_id": "p1"})
        lines = ["server line", "[kami-guard] progress t=1 downloaded=5 rate=1", "[kami-guard] check ok GPU X"]

        class Logs:
            DONE = cli.LogStream.DONE

            def __init__(self, *a, **kw):
                self.lines = mock.Mock(get=mock.Mock(side_effect=[*lines, self.DONE]))

        for verbose, expected in ((False, 2), (True, 3)):
            with mock.patch.object(cli, "LogStream", Logs), mock.patch.object(cli, "client"), \
                 mock.patch("builtins.print") as printed, mock.patch.object(cli.ui, "step"):
                cli.cmd_logs(mock.Mock(no_follow=True, tail=10, verbose=verbose), config.load_config())
            shown = [c.args[0] for c in printed.call_args_list if c.args and c.args[0] in lines]
            self.assertEqual(len(shown), expected)


class GuardIdentity(unittest.TestCase):
    def test_runpod_by_default(self):
        with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "p", "RUNPOD_API_KEY": "k"}, clear=True):
            self.assertEqual(guard.identity(), ("runpod", "p", "k"))

    def test_vast(self):
        env = {"KAMI_PROVIDER": "vast", "CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(guard.identity(), ("vast", "42", "k"))
        env = {"KAMI_PROVIDER": "vast", "VAST_CONTAINERLABEL": "C.38250", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(guard.identity()[1], "38250")

    def test_vast_deletes_its_instance(self):
        env = {"KAMI_PROVIDER": "vast", "CONTAINER_ID": "42", "CONTAINER_API_KEY": "k"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(guard.urllib.request, "urlopen") as urlopen, \
             mock.patch.object(guard, "log"):
            self.assertTrue(guard.delete_self())
        req = urlopen.call_args[0][0]
        self.assertEqual((req.get_method(), req.full_url),
                         ("DELETE", "https://console.vast.ai/api/v0/instances/42/"))
        self.assertEqual(req.get_header("Authorization"), "Bearer k")

    def test_tunnel_url_is_read_from_the_log(self):
        lines = ["noise", "[kami-guard] tunnel https://abc-def.trycloudflare.com", "more"]
        self.assertEqual(cli.tunnel_url(lines), "https://abc-def.trycloudflare.com")
        self.assertIsNone(cli.tunnel_url(["no tunnel"]))
        self.assertTrue(guard.TUNNEL_URL.search("|  https://abc-def.trycloudflare.com  |"))


class ProviderState(ProjectTestCase):
    def test_old_state_is_runpod(self):
        cfg = config.load_config()
        state = {"pod_id": "p1", "api_key": "k"}
        self.assertEqual(pod.pod_provider(state), "runpod")
        self.assertEqual(pod.pod_url(cfg, state), pod.proxy_url("p1", 8000))

    def test_vast_pod_without_url_yet(self):
        cfg = config.load_config()
        state = {"pod_id": "42", "api_key": "k", "provider": "vast"}
        self.assertIsNone(pod.pod_url(cfg, state))
        with self.assertRaises(ConfigError):
            pod.require_url(cfg, state)
        state["url"] = "https://x.trycloudflare.com"
        self.assertEqual(pod.require_url(cfg, state), "https://x.trycloudflare.com")

    def test_down_clears_provider_and_url(self):
        state = {"pod_id": "42", "api_key": "k", "provider": "vast", "url": "http://1.2.3.4:5"}
        config.save_state(state)
        pod.terminate(FakeRunPod(), state)
        saved = config.load_state()
        self.assertNotIn("provider", saved)
        self.assertNotIn("url", saved)

    def test_status_asks_the_pods_provider(self):
        config.save_state({"pod_id": "42", "provider": "vast", "url": "http://1.2.3.4:5"})
        rp = FakeRunPod({"id": "42", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 0.5})
        with mock.patch.object(cli, "client", return_value=rp) as client, \
             mock.patch.object(cli, "healthy", return_value=True), \
             mock.patch.object(cli, "server_metrics", return_value={}), \
             mock.patch("builtins.print"):
            self.assertEqual(cli.cmd_status(mock.Mock(json=True), config.load_config()), cli.STATUS_READY)
        client.assert_called_with("vast")


class FakeVast:
    name, label = "vast", "vast.ai"

    def __init__(self):
        self.created = []

    def account(self):
        return {"balance": 5.0, "spend_per_hour": 0.0}

    def create_pod(self, body):
        self.created.append(body)
        return {"id": "42", "status": "CREATED", "gpu": {"id": "RTX 4090"}, "cost": 0.4,
                "gpu_cost": 0.38, "where": "Norway, NO", "download_cost_per_gb": 0.01}

    def get_pod(self, pod_id, timeout=None):
        return {"id": "42", "provider": "vast", "status": "RUNNING", "cost": 0.4, "gpu": {"id": "RTX 4090"},
                "runtime": {"uptime": 5}, "public_ip": "1.2.3.4", "ports": {8000: 33526}}


class FakeLogs:
    def __init__(self, *args, **kw):
        pass

    def drain(self):
        return ["[kami-guard] tunnel https://abc.trycloudflare.com"]


class VastUp(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["RTX 4090"]') \
                  .replace('[defaults]\n', '[defaults]\nprovider = "vast"\n')

    def run_up(self, healthy):
        rp = FakeVast()
        args = mock.Mock(profile=None, skip_memory_check=True, max_hours=None, idle=None, budget=None,
                         secure=False, max_wait=None, poll=None, keep_on_failure=False,
                         provider=None, skip_smoke_test=False)
        with mock.patch.object(cli, "client", return_value=rp) as client, \
             mock.patch.object(cli.fit, "access_problems", return_value=[]), \
             mock.patch.object(cli.fit, "download_gb", side_effect=LookupError), \
             mock.patch.object(cli.jobs, "info", return_value={"mesh_model": "org/mesh"}), \
             mock.patch.object(cli, "healthy", side_effect=healthy), \
             mock.patch.object(cli, "LogStream", FakeLogs), \
             mock.patch.object(cli, "LogSnapshot"), \
             mock.patch.object(cli.time, "sleep"), \
             mock.patch.object(cli.mascot, "show"), \
             mock.patch("builtins.print"), mock.patch("sys.stdout"):
            cli.cmd_up(args, config.load_config())
        client.assert_called_with("vast")
        return rp, config.load_state()

    def test_up_records_the_https_tunnel(self):
        rp, state = self.run_up(lambda url, timeout=10: True)
        self.assertEqual(rp.created[0]["offer"]["gpu_name"], {"eq": "RTX 4090"})
        self.assertEqual((state["pod_id"], state["provider"]), ("42", "vast"))
        self.assertEqual(state["url"], "https://abc.trycloudflare.com")
        self.assertEqual(state["history"][-1]["provider"], "vast")
        # the boot is recorded, for the next `up` to estimate
        self.assertIsNotNone(state["history"][-1]["boot_seconds"])
        self.assertEqual(state["history"][-1]["storage"], "ephemeral")

    def test_up_falls_back_to_plain_http(self):
        # the tunnel URL is logged but never answers
        with mock.patch.object(cli, "TUNNEL_WAIT", 0):
            rp, state = self.run_up(lambda url, timeout=10: url.startswith("http://"))
        self.assertEqual(state["url"], "http://1.2.3.4:33526")


class DownloadSize(unittest.TestCase):
    FILES = {"org/mesh": (("README.md", 10), ("ckpts/a.safetensors", 20), ("ckpts/b.safetensors", 11),
                          ("ckpts/a.json", 1)),
             "org/old": (("model.bin", 7), ("x.pth", 3)),
             "org/mixed": (("model.safetensors", 5), ("pytorch_model.bin", 5))}

    def test_weight_files(self):
        self.assertEqual(len(fit.weight_files(self.FILES["org/mesh"])), 2)
        self.assertEqual(len(fit.weight_files(self.FILES["org/old"])), 2)
        # the same weights twice: the safetensors are what is loaded
        self.assertEqual(fit.weight_files(self.FILES["org/mixed"]), [("model.safetensors", 5)])

    def test_every_repo_counts(self):
        p = config.Profile("x", ["G"], image_model="org/mesh", extra_models=["org/old"])
        with mock.patch.object(fit, "repo_files", side_effect=lambda r: self.FILES[r]):
            self.assertAlmostEqual(fit.download_gb(p), 41 / 1e9)

    def test_a_repo_without_weights_is_unknown(self):
        p = config.Profile("x", ["G"], image_model="org/empty")
        with mock.patch.object(fit, "repo_files", return_value=(("README.md", 1),)), \
             self.assertRaises(LookupError):
            fit.download_gb(p)


class Prices(ProjectTestCase):
    toml = MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = ["RTX 4090", "L40S"]\ndisk_gb = 72')

    def test_session_total(self):
        q = cli.Quote("RTX 4090", stock="5 offers", gpu_h=0.40, disk_h=0.02, per_gb=0.01)
        self.assertAlmostEqual(q.session(4, 18), 0.42 * 4 + 0.18)
        self.assertAlmostEqual(q.session(4, None), 0.42 * 4)      # unknown size: GPU and disk only

    def test_runpod_quotes(self):
        cfg = config.load_config()
        rp = mock.Mock(list_gpus=mock.Mock(return_value=[
            {"id": "NVIDIA GeForce RTX 4090", "price": {"secure": 0.89}, "availability": "NONE"},
            {"id": "NVIDIA L40S", "price": {"secure": 1.09}, "availability": "HIGH"}]))
        q4090, ql40s = cli.runpod_quotes(rp, cfg, [cfg.profile(None)], "SECURE")["main"]
        self.assertEqual((q4090.stock, q4090.gpu_h, q4090.note), (None, 0.89, "no stock"))
        self.assertEqual((ql40s.stock, ql40s.gpu_h, ql40s.per_gb), ("HIGH", 1.09, 0))
        self.assertAlmostEqual(ql40s.disk_h, 72 * 0.10 / 720)

    def test_vast_quotes_search_once_per_distinct_offer(self):
        cfg = config.load_config()
        other = config.Profile("other", ["RTX 4090", "GPU nobody knows"], mesh_model="o/m2", disk_gb=72)
        rp = mock.Mock(quote=mock.Mock(side_effect=lambda f, disk: None if f["gpu_name"]["eq"] == "L40S"
                                       else {"gpu": 0.4, "disk": 0.03, "download": 0.01, "offers": 50,
                                             "hourly_max": 0.45, "download_max": 0.01}))
        quotes = cli.vast_quotes(rp, cfg, [cfg.profile(None), other], "COMMUNITY")
        self.assertEqual(rp.quote.call_count, 2)          # RTX 4090 is shared by both profiles
        q4090, ql40s = quotes["main"]
        self.assertEqual((q4090.stock, q4090.gpu_h, q4090.per_gb), ("50+ offers", 0.4, 0.01))
        self.assertEqual((ql40s.stock, ql40s.note), (None, "no offers"))
        self.assertEqual(quotes["other"][1].note, "not on vast.ai")

    def test_vast_quote_is_the_offer_up_rents(self):
        v = vast.Vast("k")
        offers = [{"dph_base": 0.40, "storage_cost": 0.36, "inet_down_cost": 0.02},
                  {"dph_base": 0.30, "storage_cost": 0.10, "inet_down_cost": 0.0}]
        with mock.patch.object(v, "search_offers", return_value=offers):
            q = v.quote({}, 72)
        self.assertEqual((q["gpu"], q["download"], q["offers"]), (0.40, 0.02, 2))
        self.assertAlmostEqual(q["disk"], 0.36 * 72 / 720)

    def test_command_prints_every_profile(self):
        cfg = config.load_config()
        rp = mock.Mock(label="RunPod", list_gpus=mock.Mock(return_value=[
            {"id": "NVIDIA L40S", "price": {"secure": 1.09}, "availability": "HIGH"}]))
        args = mock.Mock(provider="runpod", secure=True, profiles=[], hours=2)
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(fit, "download_gb", return_value=30.0), \
             mock.patch("builtins.print") as out:
            cli.cmd_prices(args, cfg)
        text = "\n".join(str(c.args[0]) if c.args else "" for c in out.call_args_list)
        self.assertIn("main", text)
        self.assertIn(f"${(1.09 + 72 * 0.10 / 720) * 2:.2f}", text)   # 2 h session, free download


class Integrity(ProjectTestCase):
    """kami.toml is checked as a whole before anything is built from it."""

    def problems(self, text):
        (self.root / "kami.toml").write_text(text, encoding="utf-8")
        with self.assertRaises(ConfigError) as ctx:
            config.load_config()
        return str(ctx.exception)

    def test_shipped_configs_are_sound(self):
        config.load_config(config.SOURCE_ROOT / "kami.toml")
        import tomllib
        self.assertEqual(config.validate(tomllib.loads(config.STARTER_CONFIG)), [])

    def test_wrong_types_are_reported_not_raised(self):
        text = self.problems(MINIMAL + 'disk_gb = "big"\n')
        self.assertIn("disk_gb must be an integer", text)
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\nport = "80a"\n'))
        self.assertIn("port must be an integer", text)

    def test_typos_suggest_the_key(self):
        text = self.problems(MINIMAL + 'image_modle = "x"\n')
        self.assertIn("unknown key 'image_modle' (did you mean 'image_model'?)", text)

    def test_key_in_the_wrong_table(self):
        text = self.problems(MINIMAL + "max_hours = 3\n")
        self.assertIn("unknown key 'max_hours' (it belongs in [defaults])", text)

    def test_gpus_as_one_string(self):
        text = self.problems(MINIMAL.replace('gpus = ["GPU A", "GPU B"]', 'gpus = "GPU A"'))
        self.assertIn("gpus must be a list of names", text)

    def test_offload_is_a_boolean(self):
        text = self.problems(MINIMAL + 'offload = "yes"\n')
        self.assertIn("offload must be true or false", text)

    def test_values_and_ranges(self):
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\ncloud = "secure"\n')
                             + "gpu_count = 0\n")
        self.assertIn("cloud must be 'SECURE' or 'COMMUNITY'", text)
        self.assertIn("gpu_count must be between 1 and 8", text)

    def test_every_problem_at_once(self):
        text = self.problems(MINIMAL + 'disk_gb = "big"\nmesh_modl = ""\n[pod]\nx = 1\n')
        self.assertIn("3 problems", text)
        self.assertIn("unknown table [pod]", text)

    def test_extends_unknown_profile(self):
        text = self.problems(MINIMAL + '\n[profiles.b]\nextends = "nope"\n')
        self.assertIn("extends unknown profile 'nope'", text)

    def test_vast_needs_known_gpus(self):
        text = self.problems(MINIMAL.replace('[defaults]\n', '[defaults]\nprovider = "vast"\n'))
        self.assertIn("'GPU A' is not in kami's GPU table", text)

    def test_cuda_as_number(self):
        (self.root / "kami.toml").write_text(MINIMAL + "min_cuda = 12.8\n", encoding="utf-8")
        self.assertEqual(config.load_config().profile(None).min_cuda, "12.8")


class BrokenConfigStillStops(ProjectTestCase):
    """A broken kami.toml must never keep `kami down` from deleting a billing pod."""

    def test_down_works_with_a_broken_config(self):
        config.save_state({"pod_id": "p1", "api_key": "k"})
        (self.root / "kami.toml").write_text(MINIMAL + 'disk_gb = "big"\n', encoding="utf-8")
        rp = FakeRunPod({"id": "p1", "status": "RUNNING", "runtime": {"uptime": 60}, "cost": 1.0})
        with mock.patch.object(cli, "client", return_value=rp), \
             mock.patch.object(cli.mascot, "show"), mock.patch("builtins.print"), \
             mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            cli.main(["--config", str(self.root / "kami.toml"), "down"])
        self.assertEqual(rp.deleted, ["p1"])
        self.assertNotIn("pod_id", config.load_state())

    def test_up_refuses_a_broken_config(self):
        (self.root / "kami.toml").write_text(MINIMAL + 'disk_gb = "big"\n', encoding="utf-8")
        with mock.patch.object(cli, "client") as client, mock.patch("builtins.print"), \
             mock.patch("sys.stdout"), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["--config", str(self.root / "kami.toml"), "up"])
        client.assert_not_called()

    def test_unexpected_errors_are_logged_with_a_billing_hint(self):
        config.save_state({"pod_id": "p1", "api_key": "k"})
        with mock.patch.object(cli, "cmd_status", side_effect=KeyError("boom")), \
             mock.patch("builtins.print"), mock.patch("sys.stdout"), mock.patch("sys.stderr"), \
             mock.patch.object(cli.ui, "error") as error, mock.patch.object(cli.ui, "hint") as hint, \
             self.assertRaises(SystemExit):
            cli.main(["--config", str(self.root / "kami.toml"), "status"])
        self.assertIn("KeyError", error.call_args[0][0])
        self.assertTrue((self.root / ".kami" / "last-error.log").exists())
        self.assertTrue(any("kami down" in str(c) for c in hint.call_args_list))


if __name__ == "__main__":
    unittest.main()


class Budget(unittest.TestCase):
    """--budget: the guard's deadline from what the pod costs, priced before creating it."""

    def plan(self, quotes, budget, provider="runpod", gb=0.0, candidates=("GPU A", "GPU B")):
        rp = mock.Mock()
        rp.name = provider
        cfg, profile = mock.Mock(), mock.Mock()
        profile.name = "main"
        fake = mock.Mock(return_value={"main": quotes})
        with mock.patch.object(cli, "runpod_quotes", fake), mock.patch.object(cli, "vast_quotes", fake), \
             mock.patch.object(cli.fit, "download_gb", return_value=gb), mock.patch.object(cli.ui, "warn"):
            return cli.budget_plan(rp, cfg, profile, [fit.Fit(c, None, None) for c in candidates],
                                   ["COMMUNITY"], budget)

    def test_hours_from_gpu_and_disk(self):
        plan = self.plan([cli.Quote("GPU A", gpu_h=1.69, disk_h=0.03)], 5.16)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["hours"], 3.0)

    def test_vast_counts_the_download_and_the_dearest_fallback_offer(self):
        q = cli.Quote("GPU A", gpu_h=0.4, disk_h=0.05, per_gb=0.01, hourly_max=0.5, per_gb_max=0.02)
        plan = self.plan([q], 3.0, provider="vast", gb=50)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["download"], 1.0)
        self.assertAlmostEqual(plan[("GPU A", "COMMUNITY")]["hours"], 4.0)    # (3 - 1) / 0.5

    def test_gpus_the_budget_cannot_keep_are_skipped(self):
        plan = self.plan([cli.Quote("GPU A", gpu_h=10.0), cli.Quote("GPU B", gpu_h=1.0),
                          cli.Quote("GPU C")], 2.0)
        self.assertEqual(list(plan), [("GPU B", "COMMUNITY")])

    def test_too_small_a_budget_creates_nothing(self):
        with self.assertRaises(ConfigError) as ctx:
            self.plan([cli.Quote("GPU A", gpu_h=10.0)], 1.0)
        self.assertIn("not enough", str(ctx.exception))
        with self.assertRaises(ConfigError):
            self.plan([cli.Quote("GPU A", gpu_h=1.0)], 0)


class Runway(unittest.TestCase):
    def warning(self, balance, spend, deadline_in=None):
        state = {"deadline": cli.time.time() + deadline_in * 3600} if deadline_in else {}
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            return " ".join(cli.runway_warning({"balance": balance, "spend_per_hour": spend}, state))

    def test_warns_before_the_guard_deadline(self):
        # seen live: $5.38 at $1.72/h lasts ~3 h, the guard allowed 7 h more
        text = self.warning(5.38, 1.72, deadline_in=7)
        self.assertIn("lasts ≈ 3h 07m", text)
        self.assertIn("before the guard's deadline", text)

    def test_quiet_when_the_balance_outlasts_the_pod(self):
        self.assertEqual(self.warning(50, 1.72, deadline_in=7), "")
        self.assertEqual(self.warning(5, 0), "")                # nothing billing

    def test_warns_when_little_is_left_without_a_guard(self):
        self.assertIn("lasts ≈ 1h 00m", self.warning(1.72, 1.72))


class Orphans(ProjectTestCase):
    def test_pods_named_by_kami_but_not_registered(self):
        rp = mock.Mock(list_pods=mock.Mock(return_value=[
            {"id": "mine", "name": "kami-main"}, {"id": "lost", "name": "kami-big", "status": "RUNNING"},
            {"id": "x", "name": "someone-else"}]))
        rp.name = "runpod"
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), \
             mock.patch.object(pod, "client", return_value=rp):
            found = pod.orphans({"pod_id": "mine", "provider": "runpod"})
        self.assertEqual([p["id"] for _, p in found], ["lost"])

    def test_providers_without_a_key_or_unreachable_are_skipped(self):
        rp = mock.Mock(list_pods=mock.Mock(side_effect=RunPodError(0, "offline")))
        with mock.patch.dict(os.environ, {"RUNPOD_API_KEY": "k"}), \
             mock.patch.object(pod, "client", return_value=rp):
            self.assertEqual(pod.orphans({}), [])


class VastBilling(unittest.TestCase):
    def test_charges_become_billing_rows_across_pages(self):
        pages = [{"results": [{"type": "instance", "source": "Instance-42", "amount": 0.5,
                               "start": 1_793_000_000, "end": 1_793_003_600}],
                  "next_token": "t"},
                 {"results": [{"type": "instance", "source": "43", "amount": 0.2,
                               "start": 1_793_000_000, "end": 1_793_000_600}]}]
        v = vast.Vast("k")
        with mock.patch.object(v, "_request", side_effect=pages) as req:
            rows = v.pod_billing("2026-10-20T00:00:00Z", bucket="day")
        self.assertEqual([r["podId"] for r in rows], ["42", "43"])
        self.assertEqual(rows[0]["timeBilledMs"], 3_600_000)
        self.assertEqual(req.call_args_list[1].kwargs["query"]["after_token"], "t")
        self.assertAlmostEqual(cli.billed_by_pod(rows)["42"]["amount"], 0.5)
