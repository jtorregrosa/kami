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
import types
import unittest
from pathlib import Path
from unittest import mock

from kami import api, catalog, cli, config, fit, guard, jobs, pod, runpod, vast, watchdog, webui

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
        for dep in (config.DINO_MODEL, config.REMBG_MODEL, *config.MESH_DEPENDENCIES):
            self.assertEqual(repos.count(dep), 1)
        image_only = config.Profile("i", ["G"], image_model="org/image")
        self.assertEqual(image_only.repos, ["org/image"])

    def test_replaced_sub_models_are_checked_instead(self):
        p = config.Profile("m", ["G"], mesh_model="org/mesh", dino_model="me/dinov3",
                           rembg_model="ZhengPeng7/BiRefNet")
        self.assertIn("me/dinov3", p.repos)
        self.assertIn("ZhengPeng7/BiRefNet", p.repos)
        self.assertNotIn(config.DINO_MODEL, p.repos)
        self.assertNotIn(config.REMBG_MODEL, p.repos)

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
        with self.assertRaisesRegex(ConfigError, "needs mesh_model, image_model or views_model"):
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
        self.assertNotIn("KAMI_DINO_MODEL", env)

    def test_min_ram_reaches_both_providers(self):
        cfg = config.load_config()
        p = config.Profile("m", ["GPU A"], mesh_model="org/mesh", min_ram_gb=48)
        self.assertEqual(pod.pod_body(cfg, p, "GPU A", None, "k")["gpu"]["minRamPerGpu"], 48)
        p0 = config.Profile("m", ["GPU A"], mesh_model="org/mesh")
        self.assertNotIn("minRamPerGpu", pod.pod_body(cfg, p0, "GPU A", None, "k")["gpu"])
        with mock.patch.object(pod.catalog, "provider_gpu", return_value="RTX_4090"), \
                mock.patch.object(pod.catalog, "find", return_value=mock.Mock(vast_max_dph=None)), \
                mock.patch.object(pod.catalog, "min_ram_mib", return_value=24000):
            p.gpu_count = 2
            self.assertEqual(pod.vast_offer(cfg, p, "GPU A", "COMMUNITY")["cpu_ram"], {"gte": 48 * 1024 * 2})
            self.assertNotIn("cpu_ram", pod.vast_offer(cfg, p0, "GPU A", "COMMUNITY"))

    def test_replaced_sub_models_reach_the_server(self):
        cfg = config.load_config()
        p = config.Profile("m", ["GPU A"], mesh_model="org/mesh", dino_model="me/dinov3",
                           rembg_model="me/birefnet")
        env = pod.pod_body(cfg, p, "GPU A", None, "k")["env"]
        self.assertEqual((env["KAMI_DINO_MODEL"], env["KAMI_REMBG_MODEL"]),
                         ("me/dinov3", "me/birefnet"))

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

    def test_dinov3_layers_aliased_for_transformers_5(self):
        layers = object()
        inner = types.SimpleNamespace(_modules={"layer": layers})
        dino = types.SimpleNamespace(_modules={"model": inner})
        pipeline = types.SimpleNamespace(image_cond_model=types.SimpleNamespace(model=dino))
        self.server.fit_transformers(pipeline)
        self.assertIs(dino.layer, layers)
        old = types.SimpleNamespace(_modules={"layer": layers})          # transformers 4
        self.server.fit_transformers(types.SimpleNamespace(
            image_cond_model=types.SimpleNamespace(model=old)))
        self.assertNotIn("layer", old.__dict__)
        self.server.fit_transformers(types.SimpleNamespace())             # no DINOv3: no-op

    def test_override_model_swaps_the_repo(self):
        class Extractor:
            def __init__(self, model_name, image_size=512):
                self.model_name, self.image_size = model_name, image_size
        module = types.SimpleNamespace(DinoV3FeatureExtractor=Extractor)
        self.server.override_model(module, "DinoV3FeatureExtractor", "me/dinov3")
        made = getattr(module, "DinoV3FeatureExtractor")(model_name="facebook/x", image_size=256)
        self.assertEqual((made.model_name, made.image_size), ("me/dinov3", 256))
        self.server.override_model(module, "Missing", "me/x")       # logged, no error

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
            data = b"".join(image for _, image in job.image_bytes)     # [(view, bytes)]
            if data == b"boom":
                raise RuntimeError("CUDA out of memory")
            with open(os.path.join(job.dir, f"{job.name}.glb"), "wb") as f:
                f.write(b"glTF" + data)
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


class WebUI(unittest.TestCase):
    """`kami ui`: the library, the local server, its guards, and the jobs it relays."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.folder = Path(tmp.name)
        for name, mtime in (("owl.glb", 100), ("owl.png", 100), ("cat.png", 200),
                            ("notes.txt", 300), ("chair.glb", 50)):
            path = self.folder / name
            path.write_bytes(b"x" * 10)
            os.utime(path, (mtime, mtime))
        (self.folder / "sub").mkdir()
        (self.folder / "sub" / "hidden.png").write_bytes(b"x")
        self.backend = mock.Mock()
        self.backend.status.return_value = {
            "pod": {"id": "p1", "status": "RUNNING", "profile": "mesh"}, "ready": True,
            "waiting": None, "served": {"mesh_model": "m", "image_model": "i"}}
        self.backend.target.return_value = ("https://pod", "pod-secret-key")
        self.backend.down.return_value = {"pod_id": "p1", "session_cost_estimate": 0.42}

    # --- the library ---

    def put(self, name, mtime=400):
        path = self.folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x" * 10)
        os.utime(path, (mtime, mtime))
        return path

    def test_the_library_is_grouped_by_object(self):
        for name, mtime in (("robot-front.png", 400), ("robot-left.png", 401), ("robot-left-2.png", 402),
                            ("robot-back-left.png", 403), ("robot-front.glb", 404), ("robot-front.3mf", 405),
                            ("tidy/vase.glb", 10)):
            self.put(name, mtime)
        os.utime(self.folder / "sub" / "hidden.png", (5, 5))
        found = webui.objects(self.folder)
        self.assertEqual([o["key"] for o in found], ["robot", "cat", "owl", "chair", "tidy/vase", "sub/hidden"])
        robot = found[0]
        self.assertEqual(robot["views"], {"front": "robot-front.png", "left": "robot-left-2.png",   # the newest take
                                          "back-left": "robot-back-left.png"})
        self.assertEqual([m["file"] for m in robot["meshes"]], ["robot-front.glb"])                 # named after its front
        self.assertEqual([m["file"] for m in robot["prints"]], ["robot-front.3mf"])
        owl = found[2]
        self.assertEqual((owl["images"], owl["meshes"][0]["file"], owl["views"]), (["owl.png"], "owl.glb", {}))
        self.assertEqual((found[4]["folder"], found[4]["name"]), ("tidy", "vase"))
        self.assertNotIn("notes", [o["name"] for o in found])

    def test_only_results_resolve(self):
        self.put("tidy/vase.glb")
        self.put("tidy/deep/too.png")
        self.put(".kami/secret.png")
        for name in ("owl.glb", "tidy/vase.glb", "sub/hidden.png"):
            self.assertIsNotNone(webui.resolve(self.folder, name), name)
        for name in ("notes.txt", "../owl.png", "tidy/../owl.glb", "tidy/deep/too.png", ".kami/secret.png",
                     "tidy\\vase.glb", "", "/owl.glb", "owl.glb/", "missing.png", None, 7):
            self.assertIsNone(webui.resolve(self.folder, name), name)

    def test_page_is_filled_once_and_names_no_file(self):
        (self.folder / "{{TOKEN}}.png").write_bytes(b"x")
        page = webui.page(self.folder, "tok123")
        self.assertIn('content="tok123"', page)
        self.assertNotIn("{{", page)
        self.assertNotIn("owl.glb", page)                     # the library comes from /api/objects
        self.assertIn(webui.MODEL_VIEWER, page)
        for stage in ("Concept", "Views", "Mesh", "Print"):
            self.assertIn(f"<b>{stage}</b>", page)

    def test_signature_changes_with_a_new_result(self):
        before = webui.signature(self.folder)
        (self.folder / "new.glb").write_bytes(b"y")
        self.assertNotEqual(webui.signature(self.folder), before)
        again = webui.signature(self.folder)
        self.put("sub/more.3mf")
        self.assertNotEqual(webui.signature(self.folder), again)

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as empty:
            self.assertEqual(webui.objects(Path(empty)), [])
        self.assertEqual(webui.results(self.folder / "missing"), [])

    def test_pictures_can_come_from_the_library(self):
        self.put("robot-front.png")
        self.put("robot-left.png")
        mesh, = webui.build_bodies({"kind": "mesh", "views": True, "images": [
            {"file": "robot-front.png", "view": "front"}, {"file": "robot-left.png", "view": "left"}]},
            folder=self.folder)
        self.assertEqual(mesh[1]["name"], "robot-front")
        self.assertEqual([i["view"] for i in mesh[1]["images"]], ["front", "left"])
        self.assertEqual(base64.b64decode(mesh[1]["images"][0]["image"]), b"x" * 10)
        views, = webui.build_bodies({"kind": "views", "angles": ["back"], "front": {"file": "robot-front.png"}},
                                    folder=self.folder)
        self.assertEqual((views[1]["name"], views[1]["params"]["views"]), ("robot", ["back"]))
        for bad in ("notes.txt", "../x.png", "owl.glb", "missing.png"):
            with self.assertRaisesRegex(ValueError, "not a picture in the library"):
                webui.build_bodies({"kind": "mesh", "images": [{"file": bad}]}, folder=self.folder)
        with self.assertRaisesRegex(ValueError, "not a picture in the library"):
            webui.build_bodies({"kind": "mesh", "images": [{"file": "robot-front.png"}]})    # no library given

    # --- requests from the page ---

    def test_build_bodies(self):
        png = base64.b64encode(b"png").decode()
        bodies = webui.build_bodies({"kind": "mesh", "images": [
            {"name": "My Owl.png", "data": png}, {"name": "b.jpg", "data": png}],
            "params": {"seed": 3, "remesh": False, "texture_size": None}})
        self.assertEqual([b["name"] for _, b in bodies], ["my-owl", "b"])
        self.assertEqual(bodies[0][1]["params"], {"seed": 3, "remesh": False})
        images = webui.build_bodies({"kind": "image", "prompt": " owl ", "count": 3,
                                     "params": {"seed": 10, "guidance": 1}})
        self.assertEqual([b["params"]["seed"] for _, b in images], [10, 11, 12])
        self.assertEqual(images[0][1]["prompt"], "owl")
        self.assertIsInstance(images[0][1]["params"]["guidance"], float)
        t2m, = webui.build_bodies({"kind": "text-to-mesh", "prompt": "owl",
                                   "params": {"image": {"seed": 1}, "mesh": {"seed": 1}}})
        self.assertEqual(t2m[1]["params"], {"image": {"seed": 1}, "mesh": {"seed": 1}})

    def test_build_bodies_refuses_bad_requests(self):
        png = base64.b64encode(b"png").decode()
        for request, error in (
                ({"kind": "video"}, "kind must be"),
                ({"kind": "mesh", "images": []}, "at least one image"),
                ({"kind": "mesh", "images": [{"name": "a.svg", "data": png}]}, "not an image"),
                ({"kind": "mesh", "images": [{"name": "a.png", "data": "%%"}]}, "base64"),
                ({"kind": "mesh", "images": [{"name": "a.png", "data": png}] * 21}, "At most"),
                ({"kind": "mesh", "images": [{"name": "a.png", "data": png}],
                  "params": {"steps": 4}}, "unknown parameter"),
                ({"kind": "mesh", "images": [{"name": "a.png", "data": png}],
                  "params": {"seed": 1.5}}, "whole number"),
                ({"kind": "image", "prompt": "  "}, "prompt"),
                ({"kind": "image", "prompt": "x", "count": 9}, "count"),
                ({"kind": "image", "prompt": "x", "params": {"negative": 3}}, "text"),
                ([], "JSON object")):
            with self.assertRaisesRegex(ValueError, error):
                webui.build_bodies(request)

    def _job(self, work, job_id):
        work.jobs[job_id] = {"id": job_id, "label": "x", "kind": "mesh", "status": "sending",
                             "position": None, "error": None, "files": [], "created": 0,
                             "finished": None}
        return work.jobs[job_id]

    def test_a_job_is_relayed_and_downloaded(self):
        work = webui.Work(self.backend, self.folder, sleep=lambda s: None)
        job = self._job(work, "w1")
        with mock.patch.object(jobs, "submit", return_value={"id": "r1", "status": "queued", "position": 1}), \
                mock.patch.object(jobs, "check", side_effect=[{"id": "r1", "status": "running"},
                                                              {"id": "r1", "status": "done", "files": ["owl.glb"]}]), \
                mock.patch.object(jobs, "download", return_value=[self.folder / "owl-2.glb"]) as download:
            work._run("w1", {"kind": "mesh"})
        self.assertEqual((job["status"], job["files"]), ("done", ["owl-2.glb"]))
        self.assertEqual(download.call_args.args[:2], ("https://pod", "pod-secret-key"))

    def test_a_failed_job_says_why(self):
        work = webui.Work(self.backend, self.folder, sleep=lambda s: None)
        job = self._job(work, "w1")
        with mock.patch.object(jobs, "submit", return_value={"id": "r1", "status": "error",
                                                             "error": "CUDA out of memory"}):
            work._run("w1", {"kind": "mesh"})
        self.assertEqual((job["status"], job["error"]), ("error", "CUDA out of memory"))
        self.backend.target.side_effect = ConfigError("No pod is active\nmore")
        job = self._job(work, "w2")
        work._run("w2", {"kind": "mesh"})
        self.assertEqual(job["error"], "No pod is active")

    def test_too_many_jobs_at_once(self):
        work = webui.Work(self.backend, self.folder)
        for i in range(webui.MAX_ACTIVE):
            work.jobs[f"w{i}"] = {"status": "running"}
        with self.assertRaisesRegex(ValueError, "Too many"):
            work.create({"kind": "image", "prompt": "x"})

    def test_pod_status_is_cached_and_never_raises(self):
        status = webui.PodStatus(self.backend, ttl=60)
        status.get()
        status.get()
        self.assertEqual(self.backend.status.call_count, 1)
        status.get(fresh=True)
        self.assertEqual(self.backend.status.call_count, 2)
        self.backend.status.side_effect = RuntimeError("provider down")
        self.assertEqual(status.get(fresh=True)["error"], "provider down")

    # --- the server ---

    def start(self):
        import threading
        from http.server import ThreadingHTTPServer
        self.port = [0]
        self.work = webui.Work(self.backend, self.folder, sleep=lambda s: None)
        status = webui.PodStatus(self.backend, ttl=0)
        server = ThreadingHTTPServer(("127.0.0.1", 0), webui.make_handler(
            self.folder, self.port, "tok", self.work, status, self.backend))
        self.port[0] = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def call(self, method, path, body=None, headers=None):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port[0], timeout=5)
        sent = {"Host": f"127.0.0.1:{self.port[0]}"}
        sent.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, path, body=data, headers=sent)
        r = conn.getresponse()
        out = (r.status, r.getheader("Content-Type"), r.read())
        conn.close()
        return out

    def act(self, path, body, **extra):
        headers = {"Content-Type": "application/json", "X-Kami-Token": "tok",
                   "Origin": f"http://127.0.0.1:{self.port[0]}"}
        headers.update(extra)
        return self.call("POST", path, body, {k: v for k, v in headers.items() if v is not None})

    def test_server_serves_results_only(self):
        self.start()
        status, kind, body = self.call("GET", "/")
        self.assertEqual((status, kind), (200, "text/html; charset=utf-8"))
        self.assertIn(b"<b>Concept</b>", body)
        self.assertEqual(self.call("GET", "/files/owl.glb")[:2], (200, "model/gltf-binary"))
        self.assertEqual(self.call("GET", "/files/sub/hidden.png")[:2], (200, "image/png"))    # one folder deep
        status, _, body = self.call("GET", "/api/objects")
        listed = json.loads(body)
        self.assertEqual((status, listed["signature"]), (200, webui.signature(self.folder)))
        self.assertIn("owl", [o["key"] for o in listed["objects"]])
        for path in ("/files/notes.txt", "/files/sub/deeper/x.png", "/files/..%2Fowl.png", "/thumbs/owl.png",
                     "/files/sub%5Chidden.png", "/files/", "/elsewhere", "/preview/x.glb", "/thumbs/notes.txt"):
            self.assertEqual(self.call("GET", path)[0], 404, path)
        self.assertEqual(self.call("GET", "/", headers={"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.call("GET", "/", headers={"Host": f"localhost:{self.port[0]}"})[0], 200)

    def test_state_never_shows_the_pod_key(self):
        self.start()
        status, _, body = self.call("GET", "/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body)
        self.assertEqual(state["pod"]["pod"]["id"], "p1")
        self.assertNotIn(b"pod-secret-key", body)
        self.assertEqual(state["print_colors"], jobs.PRINT_COLORS)
        self.assertIsInstance(state["print_hint_text"], str)   # the page shows it, to be rewritten
        self.assertEqual([p["name"] for p in state["print_presets"]], list(jobs.PRINT_PRESETS))
        self.assertEqual(set(state["print_presets"][0]), {"name", "label", "about", "text", "front", "negative"})
        self.assertIsNone(state["print_preset"])               # a Mock backend names none
        self.assertIs(state["print_hint"], True)              # a Mock backend has one

    def test_only_the_page_can_act(self):
        self.start()
        request = {"kind": "image", "prompt": "owl"}
        for extra in ({"X-Kami-Token": None},                    # no token
                      {"X-Kami-Token": "wrong"},
                      {"Origin": "https://evil.example"},        # another site
                      {"Content-Type": "text/plain"}):           # a plain form post
            self.assertEqual(self.act("/api/jobs", request, **extra)[0], 403, extra)
        rebinding = {"Host": "evil.example:80", "X-Kami-Token": "tok",
                     "Content-Type": "application/json"}
        self.assertEqual(self.call("POST", "/api/down", {}, rebinding)[0], 403)
        self.backend.down.assert_not_called()

    def test_jobs_from_the_page(self):
        self.start()
        with mock.patch.object(webui.Work, "_run"):
            status, _, body = self.act("/api/jobs", {"kind": "image", "prompt": "owl", "count": 2})
        self.assertEqual(status, 202)
        self.assertEqual(len(json.loads(body)["jobs"]), 2)
        self.assertEqual(self.act("/api/jobs", {"kind": "image", "prompt": ""})[0], 400)

    def test_no_jobs_without_a_ready_pod(self):
        self.backend.status.return_value = {"pod": None, "ready": False}
        self.start()
        status, _, body = self.act("/api/jobs", {"kind": "image", "prompt": "owl"})
        self.assertEqual(status, 409)
        self.assertIn("kami up", json.loads(body)["error"])

    def test_delete_the_pod(self):
        self.start()
        status, _, body = self.act("/api/down", {})
        self.assertEqual((status, json.loads(body)["pod_id"]), (200, "p1"))
        self.backend.down.side_effect = ConfigError("No pod is registered")
        self.assertEqual(self.act("/api/down", {})[0], 409)

    def test_cli_wires_the_command(self):
        with mock.patch.object(webui, "serve", return_value=None) as serve, \
                mock.patch.object(cli, "load_config", return_value=mock.Mock(output_dir=str(self.folder))), \
                mock.patch.object(cli, "_pod_registered", return_value=False):
            cli.main(["ui", "--no-browser", "--port", "8765"])
        folder, backend = serve.call_args.args
        self.assertEqual(folder, self.folder)
        self.assertIsInstance(backend, cli.WebBackend)
        self.assertEqual(serve.call_args.kwargs["port"], 8765)
        self.assertFalse(serve.call_args.kwargs["open_browser"])

    def test_backend_hides_the_pod_url_and_key(self):
        info = {"pod": {"id": "p1", "url": "https://p1-8000.proxy", "status": "RUNNING"},
                "ready": True, "waiting": None}
        with mock.patch.object(cli, "load_state", return_value={"pod_id": "p1", "api_key": "k"}), \
                mock.patch.object(cli, "status_info", return_value=info), \
                mock.patch.object(jobs, "info", return_value={"mesh_model": "m", "image_model": None,
                                                              "secret": "x"}) as served:
            cfg = config.load_config()
            backend = cli.WebBackend(cfg)
            first = backend.status()
            backend.status()
        self.assertNotIn("url", first["pod"])
        self.assertEqual(first["served"]["mesh_model"], "m")
        self.assertNotIn("secret", first["served"])
        served.assert_called_once()                       # cached per pod
        # which profile makes what, for the page to say how to get to another stage
        stages = {p["name"]: (p["concept"], p["views"], p["mesh"]) for p in first["profiles"]}
        self.assertEqual(set(stages), set(cfg.profiles))
        self.assertEqual(stages["views"], (True, True, False))             # pictures from text, and turns them
        self.assertEqual(stages["fronts-flux"], (True, False, False))      # pictures alone
        self.assertEqual(stages["mesh"], (False, False, True))


class Reload(unittest.TestCase):
    """`kami reload`: the server takes new code, the guard restarts it, the boot runs it."""

    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from kami import server
        self.server = server
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "kami-server.py")
        self.timer = mock.Mock()
        for patcher in (mock.patch.multiple(server, API_KEY="k", RELOAD_PATH=self.path, JOBS={},
                                            QUEUE_ORDER=[], log=mock.Mock(),
                                            COUNTS={"finished": 0, "failed": 0, "running": 0}),
                        mock.patch.object(server.threading, "Timer", self.timer)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, body, key="k"):
        return jobs._request(self.url + "/v1/reload", key, body)

    def test_new_code_is_saved_and_the_server_exits_to_load_it(self):
        status, _ = self.post({"source": "x = 1\n"})
        self.assertEqual(status, 202)
        self.assertEqual(Path(self.path).read_text(encoding="utf-8"), "x = 1\n")
        self.timer.assert_called_once()
        self.assertEqual(self.timer.call_args.args[1], os._exit)
        self.assertEqual(self.timer.call_args.kwargs["args"], (self.server.RELOAD_EXIT,))

    def test_bad_code_is_refused_and_nothing_changes(self):
        for body in ({"source": "def broken(:\n"}, {}, {"source": 3}):
            status, raw = self.post(body)
            self.assertEqual(status, 400, body)
        self.assertFalse(os.path.exists(self.path))
        self.timer.assert_not_called()

    def test_busy_server_needs_force(self):
        self.server.COUNTS["running"] = 1
        status, _ = self.post({"source": "x = 1\n"})
        self.assertEqual(status, 409)
        self.assertFalse(os.path.exists(self.path))
        self.assertEqual(self.post({"source": "x = 1\n", "force": True})[0], 202)

    def test_only_with_the_key(self):
        self.assertEqual(self.post({"source": "x = 1\n"}, key="wrong")[0], 401)
        self.assertFalse(os.path.exists(self.path))

    def test_info_says_it_can_reload(self):
        with mock.patch.object(self.server, "PIPES", mock.Mock(ready=True)):
            info = jobs.info(self.url, "k")
        self.assertTrue(info["reload"])
        self.assertIsInstance(info["started_at"], float)

    def test_boot_runs_the_reloaded_code_when_there_is_some(self):
        env = {"KAMI_RELOAD_PATH": self.path, "KAMI_SERVER_SRC": "booted = 'env'"}
        with mock.patch.dict(os.environ, env):
            scope = {}
            exec(pod.SERVER_BOOT, scope)
            self.assertEqual(scope["booted"], "env")
            Path(self.path).write_text("booted = 'reloaded'", encoding="utf-8")
            scope = {}
            exec(pod.SERVER_BOOT, scope)
            self.assertEqual(scope["booted"], "reloaded")

    def test_guard_restarts_the_server_after_a_reload(self):
        first, second = mock.Mock(), mock.Mock()
        first.wait.return_value = guard.RELOAD_EXIT
        second.wait.return_value = 3
        env = {k: v for k, v in os.environ.items()
               if k not in ("KAMI_DEADLINE", "KAMI_IDLE", "KAMI_WATCH", "KAMI_TUNNEL")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(guard.subprocess, "Popen", side_effect=[first, second]) as popen, \
                mock.patch.object(guard, "log"), mock.patch.object(guard.signal, "signal"), \
                mock.patch.object(guard.sys, "argv", ["guard", "python3", "-c", "boot"]), \
                self.assertRaises(SystemExit) as exit_:
            guard.main()
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(popen.call_args.args[0], ["python3", "-c", "boot"])
        self.assertEqual(exit_.exception.code, 3)          # any other exit ends the pod

    def test_cli_refuses_an_old_pod(self):
        state = {"pod_id": "p", "api_key": "k"}
        with mock.patch.object(cli, "load_state", return_value=state), \
                mock.patch.object(cli, "pod_url_of", return_value=self.url), \
                mock.patch.object(jobs, "info", return_value={"mesh_model": "m"}), \
                mock.patch.object(jobs, "_request") as request:
            with self.assertRaisesRegex(ConfigError, "predates"):
                cli.cmd_reload(mock.Mock(force=False), mock.Mock())
        request.assert_not_called()

    def test_cli_sends_the_code_and_waits_for_the_new_server(self):
        state = {"pod_id": "p", "api_key": "k"}
        infos = [{"reload": True, "started_at": 0},             # before
                 {"reload": True, "started_at": 0},             # old server still answering
                 {"reload": True, "started_at": 9e12}]          # the new one
        with mock.patch.object(cli, "load_state", return_value=state), \
                mock.patch.object(cli, "pod_url_of", return_value=self.url), \
                mock.patch.object(jobs, "info", side_effect=infos), \
                mock.patch.object(jobs, "_request", return_value=(202, b'{"reloading": true}')) as request, \
                mock.patch.object(cli, "healthy", side_effect=[True, False, True]), \
                mock.patch.object(cli.time, "sleep"), mock.patch.object(cli.ui, "Progress"), \
                mock.patch.object(cli.ui, "ok"):
            cli.cmd_reload(mock.Mock(force=False), mock.Mock())
        sent = request.call_args.args[2]
        self.assertIn("def make_mesh", sent["source"])
        self.assertFalse(sent["force"])


class JobProgress(unittest.TestCase):
    """Stage and progress of the running job, GET /v1/jobs, and who shows them."""

    def setUp(self):
        from kami import server
        self.server = server
        for patcher in (mock.patch.multiple(server, API_KEY="k", JOBS={}, QUEUE_ORDER=[],
                                            CURRENT=[None], log=mock.Mock()),):
            patcher.start()
            self.addCleanup(patcher.stop)

    def job(self, kind="mesh", status="running"):
        job = self.server.Job(kind, "a prompt" if kind != "mesh" else "", b"x", {}, "duck")
        job.status = status
        self.server.JOBS[job.id] = job
        return job

    # --- the server ---

    def test_stages_and_steps_add_up(self):
        job = self.job()
        self.server.CURRENT[0] = job
        self.server.stage("prepare")
        self.assertEqual(job.progress, 0.0)
        self.server.stage("structure")
        self.assertAlmostEqual(job.progress, .08)               # prepare + condition done
        self.server.step(6, 12)
        self.assertAlmostEqual(job.progress, .08 + .12 * .5)
        info = job.info()
        self.assertEqual((info["stage"], info["step"], info["steps"]), ("structure", 6, 12))
        self.assertEqual(info["stage_label"], "Sampling the sparse structure")
        self.server.stage("glb")
        self.assertAlmostEqual(job.progress, .84)
        job.status = "done"
        self.assertEqual(job.info()["progress"], 1.0)
        self.assertIsNone(job.info()["stage"])

    def test_the_cascade_shape_counts_both_passes(self):
        job = self.job()
        self.server.CURRENT[0] = job
        self.server.stage("shape", loops=2)
        base = job.progress
        self.server.step(12, 12)                                 # first pass done: half
        self.assertAlmostEqual(job.progress, base + .34 / 2)
        self.server.step(6, 12)                                  # half of the second
        self.assertAlmostEqual(job.progress, base + .34 * .75)
        self.server.step(1, 12)                                  # never backwards
        self.assertAlmostEqual(job.progress, base + .34 * .75)

    def test_text_to_mesh_gives_the_image_a_fifth(self):
        job = self.job("text-to-mesh")
        self.server.CURRENT[0] = job
        self.server.stage("image")
        self.server.step(8, 8)
        self.assertAlmostEqual(job.progress, .2)
        self.server.stage("structure")
        self.assertAlmostEqual(job.progress, .2 + .08 * .8)

    def test_no_running_job_no_effect_and_offload_has_no_share(self):
        self.server.stage("shape")
        self.server.step(1, 2)                                   # nothing runs: ignored
        job = self.job()
        self.server.CURRENT[0] = job
        self.server.stage("to_gpu")
        self.assertEqual((job.progress, job.info()["stage_label"]),
                         (0.0, "Moving the model to the GPU"))

    def test_samplers_report_their_steps_and_keep_their_bars(self):
        job = self.job()
        self.server.CURRENT[0] = job
        bars = []

        def fake_tqdm(iterable=None, *args, **kwargs):
            bars.append(kwargs.get("desc"))
            return iter(iterable)
        sampler = types.ModuleType("trellis2.pipelines.samplers.flow_euler")
        sampler.tqdm = fake_tqdm
        other = types.ModuleType("trellis2.utils.render")
        other.tqdm = fake_tqdm

        class Pipe:
            def sample_sparse_structure(self, cond):
                return [x for x in sampler.tqdm(range(4), desc="Sampling sparse structure")]

        pipe = Pipe()
        with mock.patch.dict(sys.modules, {sampler.__name__: sampler, other.__name__: other}):
            self.server.track_mesh_progress(pipe)
            self.server.track_mesh_progress(pipe)                # twice: not wrapped twice
            self.assertEqual(pipe.sample_sparse_structure("c"), [0, 1, 2, 3])
        self.assertIs(other.tqdm, fake_tqdm)                     # only the samplers
        self.assertEqual(bars, ["Sampling sparse structure"])
        self.assertEqual((job.stage, job.step, job.steps), ("structure", 4, 4))
        self.assertAlmostEqual(job.progress, .08 + .12)

    def test_listing_puts_running_then_queue_then_finished(self):
        old = self.job(status="done")
        old.finished = 1
        new = self.job(status="error")
        new.finished = 2
        second, first = self.job(status="queued"), self.job(status="queued")
        self.server.QUEUE_ORDER[:] = [first.id, second.id]
        running = self.job()
        listing = self.server.listing()
        self.assertEqual([j["id"] for j in listing["jobs"]],
                         [running.id, first.id, second.id, new.id, old.id])
        self.assertEqual((listing["running"], listing["queued"]), (1, 2))
        self.assertEqual(listing["jobs"][1]["position"], 1)

    def test_get_v1_jobs_over_http(self):
        import threading
        from http.server import ThreadingHTTPServer
        self.job()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), self.server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        listing = jobs.listing(url, "k")
        self.assertEqual(listing["jobs"][0]["name"], "duck")
        self.assertIn("progress", listing["jobs"][0])
        with self.assertRaisesRegex(ConfigError, "401"):
            jobs.listing(url, "wrong")

    def test_an_old_server_has_no_listing(self):
        with mock.patch.object(jobs, "_request", return_value=(404, b'{"error": "not found"}')):
            self.assertIsNone(jobs.listing("https://pod", "k"))

    # --- kami mesh / image ---

    def test_wait_shows_the_stage_and_a_bar(self):
        running = {"id": "j", "status": "running", "stage_label": "Sampling the shape",
                   "progress": .4, "step": 3, "steps": 12}
        with mock.patch.object(jobs.ui, "Progress") as progress, \
                mock.patch.object(jobs, "_request", return_value=(200, b'{"id": "j", "status": "done"}')):
            jobs.wait("https://pod", "k", running, "Mesh", sleep=lambda s: None)
        line = progress.return_value
        line.phase.assert_any_call("Mesh: Sampling the shape")
        line.bar.assert_called_with(.4, "step 3/12")

    # --- kami jobs ---

    def test_job_lines(self):
        listing = {"jobs": [
            {"id": "a1", "kind": "mesh", "name": "duck", "status": "running", "progress": .5,
             "stage_label": "Sampling the shape", "step": 6, "steps": 12, "started": 90},
            {"id": "b2", "kind": "image", "name": "owl", "status": "queued", "position": 1, "created": 95},
            {"id": "c3", "kind": "mesh", "name": "cat", "status": "done", "run_seconds": 61,
             "files": ["cat.glb"]},
            {"id": "d4", "kind": "mesh", "name": "dog", "status": "error", "error": "OOM\ntrace"}]}
        lines = cli.job_lines(listing, now=100)
        self.assertIn(" 50%  Sampling the shape · step 6/12 10s", lines[0])
        self.assertIn("queued #1", lines[1])
        self.assertIn("done in 1m 01s", lines[2])
        self.assertIn("cat.glb", lines[2])
        self.assertTrue(lines[3].rstrip().endswith("OOM"))

    def test_cmd_jobs(self):
        state = {"pod_id": "p", "api_key": "k"}
        listing = {"jobs": [], "running": 0, "queued": 0}
        with mock.patch.object(cli, "load_state", return_value=state), \
                mock.patch.object(cli, "pod_url_of", return_value="https://pod"), \
                mock.patch.object(jobs, "listing", return_value=listing), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            cli.cmd_jobs(mock.Mock(json=True, watch=False), mock.Mock())
        self.assertEqual(json.loads(out.getvalue()), listing)
        with mock.patch.object(cli, "load_state", return_value=state), \
                mock.patch.object(cli, "pod_url_of", return_value="https://pod"), \
                mock.patch.object(jobs, "listing", return_value=None):
            with self.assertRaisesRegex(ConfigError, "predates"):
                cli.cmd_jobs(mock.Mock(json=False, watch=False), mock.Mock())

    # --- kami ui ---

    def test_the_page_lists_every_job_on_the_pod(self):
        work = webui.Work(mock.Mock(), Path("."))
        work.jobs = {
            "w1": {"id": "w1", "label": "duck.png", "status": "running", "remote": "r1", "files": [],
                   "created": 1, "finished": None, "error": None, "position": None},
            "w2": {"id": "w2", "label": "owl", "status": "downloading", "remote": "r2", "files": [],
                   "created": 2, "finished": None, "error": None, "position": None},
            "w3": {"id": "w3", "label": "cat", "status": "sending", "remote": None, "files": [],
                   "created": 3, "finished": None, "error": None, "position": None}}
        pod = {"jobs": [
            {"id": "r1", "status": "running", "stage_label": "Sampling the shape", "progress": .5},
            {"id": "r9", "name": "from-cli", "kind": "mesh", "status": "queued", "position": 1},
            {"id": "r2", "status": "done", "progress": 1.0, "files": ["owl.png"]}]}
        listing = work.listing(pod)
        self.assertEqual([j["id"] for j in listing], ["w3", "w1", "pod-r9", "w2"])
        self.assertEqual((listing[1]["label"], listing[1]["progress"]), ("duck.png", .5))
        self.assertTrue(listing[2]["external"])
        self.assertEqual(listing[3]["status"], "downloading")    # done there, still coming here
        self.assertEqual(len(work.listing(None)), 3)
        self.assertEqual(len(work.listing(mock.Mock())), 3)

    def test_pod_jobs_cache_and_failures(self):
        backend = mock.Mock()
        backend.jobs.return_value = {"jobs": []}
        cache = webui.PodJobs(backend, ttl=60)
        self.assertIsNone(cache.get(ready=False))
        self.assertEqual(cache.get(ready=True), {"jobs": []})
        cache.get(ready=True)
        backend.jobs.assert_called_once()
        backend.jobs.side_effect = ConfigError("unreachable")
        self.assertIsNone(webui.PodJobs(backend).get(ready=True))


class GuardDeletesItself(unittest.TestCase):
    """The pod-scoped RunPod key was refused by api.runpod.io/v2 (403) and rest.runpod.io
    did not resolve on a real pod: GraphQL goes first, and its errors count as failures."""

    def answer(self, body):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = json.dumps(body).encode()
        return resp

    def run_guard(self, answers):
        env = {"RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "scoped", "KAMI_PROVIDER": "runpod"}
        with mock.patch.dict(os.environ, env), mock.patch.object(guard, "log"), \
                mock.patch.object(guard.urllib.request, "urlopen", side_effect=answers) as urlopen:
            done = guard.delete_self()
        return done, [c.args[0] for c in urlopen.call_args_list]

    def test_graphql_terminate_first(self):
        done, reqs = self.run_guard([self.answer({"data": {"podTerminate": None}})])
        self.assertTrue(done)
        self.assertEqual(reqs[0].full_url, "https://api.runpod.io/graphql")
        query = json.loads(reqs[0].data)["query"]
        self.assertIn('podTerminate(input: {podId: "pod1"})', query)
        self.assertEqual(reqs[0].get_header("Authorization"), "Bearer scoped")

    def test_a_graphql_error_tries_the_next_way(self):
        done, reqs = self.run_guard([self.answer({"errors": [{"message": "not allowed"}]}),
                                     OSError("Name or service not known"),
                                     self.answer({})])
        self.assertTrue(done)
        self.assertEqual([r.get_method() for r in reqs], ["POST", "DELETE", "DELETE"])
        self.assertEqual(reqs[2].full_url, "https://api.runpod.io/v2/pods/pod1")

    def test_every_way_failing_is_reported(self):
        done, reqs = self.run_guard([OSError("down")] * len(guard.ACTIONS["runpod"]))
        self.assertFalse(done)
        self.assertEqual(len(reqs), len(guard.ACTIONS["runpod"]))
        self.assertIn('podStop(input: {podId: "pod1"})', json.loads(reqs[3].data)["query"])


class MultiView(unittest.TestCase):
    """Several views of one object: TRELLIS.2's tuning-free multiview, Hunyuan3D-2(mv)."""

    def setUp(self):
        from kami import server
        self.server = server
        patcher = mock.patch.multiple(server, CURRENT=[None], log=mock.Mock())
        patcher.start()
        self.addCleanup(patcher.stop)

    def b64(self, data=b"png"):
        return base64.b64encode(data).decode()

    def backend(self, backend="trellis2", mv=False):
        return mock.patch.multiple(self.server, MESH_BACKEND=backend, HUNYUAN_MV=mv,
                                   MESH_MODEL="tencent/Hunyuan3D-2" if backend == "hunyuan" else "m")

    # --- the job body ---

    def test_one_image_is_one_unnamed_view(self):
        with self.backend():
            self.assertEqual(self.server.mesh_views({"image": self.b64()}), [(None, b"png")])

    def test_trellis_takes_any_views_up_to_the_limit(self):
        with self.backend():
            views = self.server.mesh_views({"images": [{"image": self.b64(b"a")}, {"image": self.b64(b"b"), "view": "side"}]})
            self.assertEqual(views, [(None, b"a"), ("side", b"b")])
            with self.assertRaisesRegex(ValueError, "at most 8"):
                self.server.mesh_views({"images": [{"image": self.b64()}] * 9})

    def test_hunyuan_single_refuses_views(self):
        with self.backend("hunyuan"):
            with self.assertRaisesRegex(ValueError, "one image.*Hunyuan3D-2mv"):
                self.server.mesh_views({"images": [{"image": self.b64()}] * 2})

    def test_hunyuan_mv_names_the_views(self):
        with self.backend("hunyuan", mv=True):
            views = self.server.mesh_views({"images": [{"image": self.b64(b"a"), "view": "back"},
                                                       {"image": self.b64(b"b")}, {"image": self.b64(b"c")}]})
            self.assertEqual([v for v, _ in views], ["back", "front", "left"])
            for bad, why in (([{"image": self.b64(), "view": "top"}], "unknown view"),
                             ([{"image": self.b64(), "view": "front"}] * 2, "once"),
                             ([{"image": self.b64()}] * 5, "at most 4")):
                with self.assertRaisesRegex(ValueError, why):
                    self.server.mesh_views({"images": bad})

    def test_bad_bodies(self):
        with self.backend():
            for body, why in (({"images": []}, "needs an image"), ({"images": "x"}, "list"),
                              ({"images": [{"image": "%%"}]}, "base64"), ({}, "needs an image")):
                with self.assertRaisesRegex(ValueError, why):
                    self.server.mesh_views(body)

    def test_submit_checks_the_mode_and_keeps_the_views(self):
        with self.backend(), mock.patch.multiple(self.server, JOBS={}, QUEUE_ORDER=[], QUEUE=mock.Mock(),
                                                 PIPES=mock.Mock(mesh=object(), image=None)):
            with self.assertRaisesRegex(ValueError, "multiview_mode"):
                self.server.submit({"kind": "mesh", "image": self.b64(), "params": {"multiview_mode": "x"}})
            job = self.server.submit({"kind": "mesh", "images": [{"image": self.b64()}] * 3,
                                      "params": {"multiview_mode": "multidiffusion"}})
            self.assertEqual(job.info()["views"], 3)

    def test_info_tells_what_the_pod_takes(self):
        with self.backend("hunyuan", mv=True):
            self.assertEqual(self.server.view_limits(), (4, ("front", "left", "back", "right")))
        with self.backend("hunyuan"):
            self.assertEqual(self.server.view_limits(), (1, None))
        with self.backend():
            self.assertEqual(self.server.view_limits(), (8, None))

    # --- TRELLIS.2 ---

    def fake_pipe(self):
        """A pipeline whose samplers record which view each step saw."""
        seen = []

        class Sampler:
            def _inference_model(self, flow, x_t, t, cond=None, **kwargs):
                seen.append((list(cond), kwargs.get("neg_cond")))
                return cond[0] * 10

        class Pipe:
            sparse_structure_sampler = Sampler()
            shape_slat_sampler = Sampler()
            tex_slat_sampler = shape_slat_sampler            # shared: patched once

            def get_cond(self, images, resolution, include_neg_cond=True):
                return {"cond": list(images), "neg_cond": ["neg"] * len(images)}

        return Pipe(), seen

    def test_stochastic_gives_each_step_one_view(self):
        pipe, seen = self.fake_pipe()
        with self.server.trellis_multiview(pipe, [1, 2, 3], "stochastic"):
            cond = pipe.get_cond(["ignored"], 512)
            self.assertEqual(cond, {"cond": [1, 2, 3], "neg_cond": ["neg"]})
            sampler = pipe.sparse_structure_sampler
            outs = [sampler._inference_model("flow", "x", .5, cond["cond"], neg_cond="n") for _ in range(4)]
        self.assertEqual(outs, [10, 20, 30, 10])
        self.assertEqual([s[0] for s in seen], [[1], [2], [3], [1]])
        self.assertEqual(seen[0][1], "n")                        # guidance arguments pass through
        self.assertNotIn("_inference_model", vars(pipe.sparse_structure_sampler))   # restored
        self.assertNotIn("get_cond", vars(pipe))

    def test_multidiffusion_averages_every_view(self):
        pipe, seen = self.fake_pipe()
        with self.server.trellis_multiview(pipe, [1, 2, 3], "multidiffusion"):
            out = pipe.shape_slat_sampler._inference_model("flow", "x", .5, [1, 2, 3])
        self.assertEqual(out, 20)
        self.assertEqual(len(seen), 3)

    def test_restored_after_a_failure_and_bad_mode(self):
        pipe, _ = self.fake_pipe()
        with self.assertRaises(RuntimeError):
            with self.server.trellis_multiview(pipe, [1, 2], "stochastic"):
                raise RuntimeError("OOM")
        self.assertNotIn("_inference_model", vars(pipe.sparse_structure_sampler))
        with self.assertRaisesRegex(ValueError, "multiview_mode"):
            with self.server.trellis_multiview(pipe, [1, 2], "best"):
                pass

    def test_make_mesh_trellis_with_views(self):
        pipe = mock.Mock()
        pipe.preprocess_image.side_effect = lambda im: "pre-" + im
        pipe.run.side_effect = RuntimeError("stop after run")    # what follows needs o_voxel
        fake_o_voxel = types.ModuleType("o_voxel")
        with mock.patch.dict(sys.modules, {"o_voxel": fake_o_voxel}), \
                mock.patch.object(self.server, "PIPES", mock.Mock(mesh=pipe)), \
                mock.patch.object(self.server, "trellis_multiview") as multiview:
            with self.assertRaisesRegex(RuntimeError, "stop after run"):
                self.server.make_mesh_trellis([(None, "a"), (None, "b")],
                                              {"multiview_mode": "stochastic", "seed": 3}, "out.glb")
        multiview.assert_called_once_with(pipe, ["pre-a", "pre-b"], "stochastic")
        args, kwargs = pipe.run.call_args
        self.assertEqual(args, ("pre-a",))
        self.assertFalse(kwargs["preprocess_image"])

    # --- the client ---

    def test_view_args(self):
        self.assertEqual(jobs.view_arg("front=a.png"), ("front", "a.png"))
        self.assertEqual(jobs.view_arg("Back=C:\\x\\b.png"), ("back", "C:\\x\\b.png"))
        self.assertEqual(jobs.view_arg("C:\\x\\a=b.png"), (None, "C:\\x\\a=b.png"))
        self.assertEqual(jobs.view_arg("a.png"), (None, "a.png"))

    def cmd_mesh(self, images, served, **flags):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        paths = []
        for name in images:
            view, _, file = name.rpartition("=")
            path = Path(tmp.name) / file
            path.write_bytes(b"\x89PNG" + file.encode())
            paths.append(f"{view}={path}" if view else str(path))
        args = mock.Mock(images=paths, prompt=None, views=flags.get("views", False),
                         view_mode=flags.get("view_mode"), out_dir=tmp.name, seed=None,
                         texture_size=None, decimation_target=None, no_remesh=False, no_texture=False)
        with mock.patch.object(cli, "ensure_pod", return_value={"profile": "p"}), \
                mock.patch.object(cli, "job_target", return_value=("u", "k", served)), \
                mock.patch.object(cli, "run_job") as run_job, mock.patch.object(cli.ui, "hint"):
            cli.cmd_mesh(args, mock.Mock())
        return run_job

    def test_kami_mesh_views_is_one_job(self):
        run_job = self.cmd_mesh(["a.png", "b.png", "c.png"], {"mesh_model": "m", "max_views": 8},
                                views=True, view_mode="multidiffusion")
        run_job.assert_called_once()
        body = run_job.call_args.args[2]
        self.assertEqual(len(body["images"]), 3)
        self.assertEqual(body["params"]["multiview_mode"], "multidiffusion")
        self.assertEqual(body["name"], "a")

    def test_named_views_imply_views(self):
        run_job = self.cmd_mesh(["front=a.png", "back=b.png"],
                                {"mesh_model": "m", "max_views": 4, "view_names": ["front", "left", "back", "right"]})
        body = run_job.call_args.args[2]
        self.assertEqual([i["view"] for i in body["images"]], ["front", "back"])

    def test_without_views_one_job_per_image(self):
        run_job = self.cmd_mesh(["a.png", "b.png"], {"mesh_model": "m", "max_views": 8})
        self.assertEqual(run_job.call_count, 2)

    def test_views_refused_where_the_pod_cannot(self):
        with self.assertRaisesRegex(ConfigError, "one image"):
            self.cmd_mesh(["a.png", "b.png"], {"mesh_model": "m", "max_views": 1}, views=True)
        with self.assertRaisesRegex(ConfigError, "predates"):
            self.cmd_mesh(["a.png", "b.png"], {"mesh_model": "m"}, views=True)
        with self.assertRaisesRegex(ConfigError, "two or more"):
            self.cmd_mesh(["a.png"], {"mesh_model": "m", "max_views": 8}, views=True)

    def test_the_page_sends_views_as_one_job(self):
        image = {"name": "a.png", "data": self.b64()}
        bodies = webui.build_bodies({"kind": "mesh", "views": True,
                                     "images": [dict(image, view="front"), dict(image, name="b.png")],
                                     "params": {"multiview_mode": "multidiffusion"}})
        self.assertEqual(len(bodies), 1)
        label, body = bodies[0]
        self.assertIn("2 views", label)
        self.assertEqual([i["view"] for i in body["images"]], ["front", None])
        self.assertEqual(len(webui.build_bodies({"kind": "mesh", "images": [image, image]})), 2)
        for bad in ({"views": True, "images": [dict(image, view="top")]},
                    {"images": [image], "params": {"multiview_mode": "best"}}):
            with self.assertRaises(ValueError):
                webui.build_bodies({"kind": "mesh", **bad})


def _has_pillow():
    try:
        import PIL.Image  # noqa: F401
        return True
    except ImportError:
        return False


class Views(unittest.TestCase):
    """`kami views`: profiles, the pod env, the server's views jobs, the CLI and the page."""

    def load(self, extra):
        text = MINIMAL + extra
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kami.toml"
            path.write_text(text, encoding="utf-8")
            return config.load_config(path)

    QWEN = """
[profiles.vq]
image_model = "Tongyi-MAI/Z-Image-Turbo"
views_model = "Qwen/Qwen-Image-Edit-2511"
views_lora = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/lora.safetensors"
views_defaults = { steps = 30, lora_scale = 0.8 }
gpus = ["H100 PCIe"]
"""

    def test_a_views_profile_downloads_its_model_and_lora_and_tells_the_pod(self):
        cfg = self.load(self.QWEN)
        p = cfg.profiles["vq"]
        self.assertEqual(p.repos, ["Tongyi-MAI/Z-Image-Turbo", "Qwen/Qwen-Image-Edit-2511",
                                   "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/lora.safetensors"])
        env = pod.server_spec(cfg, p, None, "key")[0]
        self.assertEqual(env["KAMI_VIEWS_MODEL"], "Qwen/Qwen-Image-Edit-2511")
        self.assertTrue(env["KAMI_VIEWS_LORA"].endswith("/lora.safetensors"))
        self.assertEqual(json.loads(env["KAMI_VIEWS_DEFAULTS"]), {"steps": 30, "lora_scale": 0.8})
        self.assertNotIn("KAMI_VIEWS_MODEL", pod.server_spec(cfg, cfg.profiles["main"], None, "key")[0])

    def test_a_views_profile_needs_no_other_model_but_a_well_formed_lora(self):
        self.load('\n[profiles.v]\nviews_model = "diffusers/FLUX.2-dev-bnb-4bit"\ngpus = ["L40S"]\n')
        with self.assertRaisesRegex(ConfigError, "org/name/file.safetensors"):
            self.load('\n[profiles.v]\nviews_model = "a/b"\nviews_lora = "a/lora"\ngpus = ["L40S"]\n')
        with self.assertRaisesRegex(ConfigError, "views_lora needs a views_model"):
            self.load('\n[profiles.v]\nimage_model = "a/b"\nviews_lora = "a/b/c.safetensors"\ngpus = ["L40S"]\n')
        with self.assertRaisesRegex(ConfigError, "unknown parameter 'width'"):
            self.load('\n[profiles.v]\nviews_model = "a/b"\nviews_defaults = { width = 9 }\ngpus = ["L40S"]\n')

    def test_a_lora_file_counts_alone_in_the_download(self):
        files = (("lora.safetensors", 300), ("other.safetensors", 999), ("demo.gif", 50))
        with mock.patch.object(fit, "repo_files", lambda repo: files):
            self.assertEqual(fit.entry_files("org/name/lora.safetensors"), [("lora.safetensors", 300)])

    def test_the_camera_lora_prompt_and_the_view_names(self):
        from kami import server
        p = {"elevation": "eye-level shot", "distance": "medium shot"}
        self.assertEqual(server.view_prompt("back", p), "<sks> back view eye-level shot medium shot")
        # kami's (Hunyuan3D-2mv's) left is the camera on your left: the LoRAs' right side
        self.assertEqual(server.view_prompt("left", p), "<sks> right side view eye-level shot medium shot")
        self.assertEqual(server.view_prompt("front-left", p),
                         "<sks> front-right quarter view eye-level shot medium shot")
        self.assertEqual(tuple(server.VIEW_ANGLES), jobs.VIEW_ANGLES)
        self.assertEqual(server.lora_parts("org/name/dir/l.safetensors"), ("org/name", "dir/l.safetensors"))

    def test_views_params_are_checked(self):
        from kami import server
        with mock.patch.multiple(server, PIPES=mock.Mock(image=None), VIEWS_FAMILY="qwen-edit",
                                 VIEWS_MODEL="Qwen/Qwen-Image-Edit-2511"):
            server.check_views_params({"views": ["left", "back"]}, has_front=True)
            for params, error in (({"views": ["up"]}, "views must be"), ({"views": []}, "views must be"),
                                  ({"views": ["left", "left"]}, "once"),
                                  ({"elevation": 3}, "elevation must be text")):
                with self.assertRaisesRegex(ValueError, error):
                    server.check_views_params(params, has_front=True)
            # Qwen-Edit cannot make the front from text without an image model
            with self.assertRaisesRegex(ValueError, "give a front image"):
                server.check_views_params({}, has_front=False)
        with mock.patch.multiple(server, PIPES=mock.Mock(image=None), VIEWS_FAMILY="flux2"):
            server.check_views_params({}, has_front=False)      # FLUX.2 makes it itself

    def test_a_views_job_turns_the_camera_around_the_front(self):
        from kami import server
        calls = []

        class Picture:
            mode, size = "RGB", (512, 512)

            def convert(self, mode):
                return self

            def resize(self, size, *a):
                return self

            def save(self, path):
                open(path, "wb").close()

        class Pipe:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                kwargs["callback_on_step_end"](self, 0, 0, {})
                return types.SimpleNamespace(images=[Picture()])

            def enable_lora(self):
                calls.append("lora on")

            def disable_lora(self):
                calls.append("lora off")

            def set_adapters(self, names, adapter_weights):
                calls.append(("scale", adapter_weights[0]))
        torch = types.ModuleType("torch")
        torch.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda s: ("gen", s))
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"torch": torch}), \
                mock.patch.multiple(server, PIPES=mock.Mock(image=None, views=Pipe()),
                                    VIEWS_FAMILY="flux2", VIEWS_LORA="o/n/l.safetensors",
                                    on_white=lambda im: im, editing_size=lambda im: im,
                                    log=mock.Mock()):
            job = server.Job("views", "a ceramic owl", None, {"seed": 7, "views": ["left", "back"]}, "owl")
            job.dir = tmp
            server.CURRENT[0] = job
            try:
                server.make_views(job)
            finally:
                server.CURRENT[0] = None
        self.assertEqual(job.files, ["owl-front.png", "owl-left.png", "owl-back.png"])
        front, left, back = [c for c in calls if isinstance(c, dict)]
        self.assertNotIn("image", front)                        # FLUX.2 made the front from text
        self.assertIn("a ceramic owl, front view", front["prompt"])
        self.assertEqual(calls[0], "lora off")                  # no camera LoRA for the front
        self.assertIn(("scale", 0.9), calls)
        self.assertEqual(left["prompt"], "<sks> right side view eye-level shot medium shot")
        self.assertEqual(back["prompt"], "<sks> back view eye-level shot medium shot")
        self.assertEqual((back["width"], back["height"], back["generator"]), (512, 512, ("gen", 7)))
        self.assertEqual(len(left["image"]), 1)

    def test_an_old_jinja2_is_upgraded_on_boot(self):
        from importlib import metadata
        from kami import server
        for have, installs in (("3.1.5", True), ("3.1.6", False), ("3.2", False)):
            with self.subTest(have=have), mock.patch.object(metadata, "version", return_value=have), \
                    mock.patch("subprocess.run") as run, mock.patch.object(server, "log"):
                server.ensure_packages()
            self.assertEqual(run.called, installs)
            if installs:
                self.assertEqual(run.call_args[0][0][-1], "jinja2>=3.1.6")

    def test_qwen_edit_is_called_with_true_cfg(self):
        from kami import server
        seen = {}

        class Pipe:
            def __call__(self, prompt, image, num_inference_steps, true_cfg_scale, negative_prompt,
                         guidance_scale, callback_on_step_end, generator=None):
                seen.update(true_cfg_scale=true_cfg_scale, negative=negative_prompt)
                return types.SimpleNamespace(images=["picture"])
        with mock.patch.dict(sys.modules, {"torch": types.ModuleType("torch")}), \
                mock.patch.multiple(server, PIPES=mock.Mock(views=Pipe()), VIEWS_FAMILY="qwen-edit"):
            out = server.views_call("<sks> back view", {"steps": 4, "guidance": 4.0}, "front")
        self.assertEqual((out, seen), ("picture", {"true_cfg_scale": 4.0, "negative": " "}))

    def test_the_cli_names_the_views_for_kami_mesh(self):
        saved = [Path("out/owl-front.png"), Path("out/owl-left-2.png"), Path("out/owl-back.png"),
                 Path("out/owl-front-left.png")]
        self.assertEqual(cli.mesh_command(saved),
                         ("kami mesh front=out/owl-front.png left=out/owl-left-2.png back=out/owl-back.png "
                          "front-left=out/owl-front-left.png").replace("/", os.sep))

    def test_a_front_picture_named_front_gives_its_name_without_it(self):
        self.assertEqual(jobs.views_name("owl-front"), "owl")
        self.assertEqual(jobs.views_name("Owl_front_left"), "owl")
        self.assertEqual(jobs.views_name("owl"), "owl")
        self.assertEqual(jobs.views_name("front"), "front")

    def test_the_page_builds_a_views_job(self):
        (label, body), = webui.build_bodies({"kind": "views", "prompt": " an owl ",
                                             "angles": ["left", "back", "left"],
                                             "params": {"seed": 3, "lora_scale": None}})
        self.assertEqual(body, {"kind": "views", "prompt": "an owl", "name": "an-owl",
                                "params": {"seed": 3, "views": ["left", "back"]}})
        png = base64.b64encode(b"png").decode()
        (_, body), = webui.build_bodies({"kind": "views", "front": {"name": "owl.png", "data": png}})
        self.assertEqual((body["image"], body["name"]), (png, "owl"))
        with self.assertRaisesRegex(ValueError, "angles must be"):
            webui.build_bodies({"kind": "views", "prompt": "x", "angles": ["up"]})


class Pixal3D(unittest.TestCase):
    """Pixal3D as a mesh backend: its posed cameras, the framing of the views, the
    profile and what reaches the pod."""

    def test_the_cameras_are_those_of_its_example_rig(self):
        from kami import server
        # transforms.json of Pixal3D's assets/mv_images/example (rows, rounded)
        d = server.PIXAL_DISTANCE
        example = {0: [[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]],
                   90: [[0, 0, 1, d], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                   180: [[-1, 0, 0, 0], [0, 0, 1, d], [0, 1, 0, 0], [0, 0, 0, 1]],
                   270: [[0, 0, -1, -d], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]}
        for azimuth, matrix in example.items():
            got = server.pixal_camera(azimuth)
            for row, want in zip(got, matrix):
                for a, b in zip(row, want):
                    self.assertAlmostEqual(a, b, places=6, msg=f"azimuth {azimuth}")
        # kami's left (the object facing the picture's left) is its azimuth 90
        self.assertEqual(server.PIXAL_AZIMUTHS["left"], 90)
        self.assertEqual(tuple(sorted(server.PIXAL_VIEW_ORDER)), tuple(sorted(jobs.VIEW_ANGLES)))

    def test_the_distance_matches_its_inference_script(self):
        import math
        from kami import server
        # inference.py's distance_from_fov for fov 0.2 rad, 512 px, mesh_scale 1
        f_pixels = (16.0 / math.tan(0.1)) * 512 / 32.0
        self.assertAlmostEqual(server.pixal_distance(0.2), f_pixels * -0.5 / -256.0)

    @unittest.skipUnless(_has_pillow(), "needs Pillow")
    def test_views_get_one_framing_each_centred_on_its_object(self):
        from PIL import Image
        from kami import server
        wide = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        wide.paste((255, 0, 0, 255), (50, 40, 250, 140))      # 200 x 100 object, off centre
        tall = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
        tall.paste((0, 255, 0, 255), (140, 20, 180, 170))     # 40 x 150
        a, b = server.same_framing([wide, tall])
        self.assertEqual(a.size, b.size)
        self.assertEqual(a.size, (220, 220))                  # the largest extent x 1.1
        for square in (a, b):
            l, t, r, bottom = square.getchannel("A").getbbox()
            self.assertLessEqual(abs((l + r) / 2 - square.width / 2), 1)
            self.assertLessEqual(abs((t + bottom) / 2 - square.height / 2), 1)

    def test_posed_views_need_a_front(self):
        from kami import server
        png = base64.b64encode(b"png").decode()
        with mock.patch.multiple(server, MESH_BACKEND="pixal3d"):
            self.assertEqual(server.view_limits()[0], 8)
            views = server.mesh_views({"images": [{"image": png}, {"image": png, "view": "back-left"}]})
            self.assertEqual([v for v, _ in views], ["front", "back-left"])
            with self.assertRaisesRegex(ValueError, "front"):
                server.mesh_views({"images": [{"image": png, "view": "left"},
                                              {"image": png, "view": "back"}]})

    def test_a_pixal3d_profile(self):
        text = MINIMAL + """
[profiles.px]
mesh_model = "TencentARC/Pixal3D"
gpus = ["RTX 4090"]
mesh_defaults = { resolution = 1024, fov = 15.0 }
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kami.toml"
            path.write_text(text, encoding="utf-8")
            cfg = config.load_config(path)
        p = cfg.profiles["px"]
        self.assertEqual(p.mesh_backend, "pixal3d")
        self.assertEqual(p.repos, ["TencentARC/Pixal3D", config.DINO_MODEL, config.REMBG_MODEL])
        env = pod.server_spec(cfg, p, None, "key")[0]
        self.assertEqual(env["KAMI_MESH_BACKEND"], "pixal3d")

    def test_the_cli_takes_any_named_view(self):
        self.assertEqual(jobs.view_arg("back-left=a.png"), ("back-left", "a.png"))
        self.assertEqual(jobs.view_arg("top=a.png"), (None, "top=a.png"))


@unittest.skipUnless(_has_pillow(), "needs Pillow")
class UploadedImages(unittest.TestCase):
    """server.open_image: the formats people upload, as RGB / RGBA for the pipelines."""

    def encode(self, image, fmt, **kwargs):
        buf = io.BytesIO()
        image.save(buf, fmt, **kwargs)
        return buf.getvalue()

    def open(self, data):
        from kami import server
        return server.open_image(data)

    def test_common_formats_open_as_rgb(self):
        from PIL import Image, features
        photo = Image.new("RGB", (64, 48), (200, 120, 40))
        formats = ["PNG", "JPEG", "BMP", "TIFF", "GIF"]
        formats += [f for f, feature in (("WEBP", "webp"), ("AVIF", "avif"))
                    if features.check(feature)]
        for fmt in formats:
            with self.subTest(fmt=fmt):
                image = self.open(self.encode(photo, fmt))
                self.assertEqual((image.mode, image.size), ("RGB", (64, 48)))

    def test_a_real_cut_out_keeps_its_alpha(self):
        from PIL import Image
        cut = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        cut.paste((255, 220, 0, 255), (10, 10, 30, 30))
        self.assertEqual(self.open(self.encode(cut, "PNG")).mode, "RGBA")
        # a palette PNG with a transparent colour too
        self.assertEqual(self.open(self.encode(cut.convert("P"), "PNG", transparency=0)).mode,
                         "RGBA")

    def test_an_almost_opaque_alpha_is_dropped_so_the_background_is_removed(self):
        from PIL import Image
        # like the WebP that made a mesh of its whole square: alpha 143..255, nothing clear
        image = Image.new("RGBA", (100, 100), (250, 230, 170, 255))
        image.putpixel((0, 0), (250, 230, 170, 143))
        self.assertEqual(self.open(self.encode(image, "PNG")).mode, "RGB")

    def test_sixteen_bit_and_grayscale_become_rgb(self):
        from PIL import Image
        deep = Image.new("I;16", (8, 8), 65535)
        image = self.open(self.encode(deep, "PNG"))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.open(self.encode(Image.new("L", (8, 8), 9), "PNG")).mode, "RGB")

    def test_a_phone_photo_is_turned_as_its_exif_says(self):
        from PIL import Image
        photo = Image.new("RGB", (60, 20), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6                                # orientation: rotate 90 CW
        image = self.open(self.encode(photo, "JPEG", exif=exif.tobytes()))
        self.assertEqual(image.size, (20, 60))

    def test_the_client_offers_what_the_server_reads(self):
        for suffix in (".png", ".jpg", ".webp", ".avif", ".heic", ".tiff", ".bmp", ".gif"):
            self.assertIn(suffix, jobs.IMAGE_SUFFIXES)


class Hunyuan(unittest.TestCase):
    """Hunyuan3D-2 profiles: licence, repos, pod env."""

    def test_an_image_pillow_cannot_read_is_decoded_with_opencv(self):
        from kami import server
        pil_image = types.SimpleNamespace(fromarray=lambda pixels, mode=None: ("pil", pixels, mode))
        pil = types.ModuleType("PIL")
        pil.Image = pil_image

        class Pixels:
            dtype, ndim, shape = "uint8", 3, (4, 4, 4)
        cv2 = types.SimpleNamespace(
            IMREAD_UNCHANGED=-1, COLOR_BGRA2RGBA="bgra>rgba", COLOR_BGR2RGB="bgr>rgb",
            imdecode=lambda buf, flag: Pixels(), cvtColor=lambda px, code: ("rgb", code))
        numpy = types.SimpleNamespace(uint8="uint8", frombuffer=lambda data, dtype: data)
        broken = ValueError("too many values to unpack (expected 5)")
        modules = {"PIL": pil, "PIL.Image": pil_image, "cv2": cv2, "numpy": numpy}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(server, "log"):
            image = server.opencv_image(b"RIFF....WEBP", broken)
        self.assertEqual(image, ("pil", ("rgb", "bgra>rgba"), None))

        cv2.imdecode = lambda buf, flag: None          # OpenCV cannot either: a clear error
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "cannot read the image"):
                server.opencv_image(b"junk", broken)

    def test_only_the_images_hunyuan_pipeline_code_is_trusted(self):
        from kami import server
        seen = []

        class DiffusionPipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                seen.append(kwargs.get("trust_remote_code"))
                return cls
        fake = types.ModuleType("diffusers")
        fake.DiffusionPipeline = DiffusionPipeline
        with mock.patch.dict(sys.modules, {"diffusers": fake}), \
                mock.patch.object(server.os.path, "realpath", side_effect=lambda p: p):
            with server.trust_hunyuan_pipelines():
                DiffusionPipeline.from_pretrained(
                    "ckpt", custom_pipeline="/opt/Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint")
                DiffusionPipeline.from_pretrained("ckpt", custom_pipeline="/tmp/other")
                DiffusionPipeline.from_pretrained("ckpt")
            DiffusionPipeline.from_pretrained(
                "ckpt", custom_pipeline="/opt/Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint")
        self.assertEqual(seen, [True, None, None, None])

    def test_dinov2_weights_get_the_transformers_5_names(self):
        from kami import server

        class Halves(list):
            def chunk(self, n, dim=0):
                half = len(self) // n
                return Halves(self[:half]), Halves(self[half:])
        prefix = "main_image_encoder.model.encoder.layer.0."
        old = {prefix + "attention.attention.query.weight": "q",
               prefix + "attention.output.dense.bias": "o",
               prefix + "mlp.weights_in.weight": Halves([1, 2, 3, 4]),
               prefix + "mlp.weights_out.weight": "d",
               prefix + "norm1.weight": "n"}
        expected = [prefix + "attention.q_proj.weight", prefix + "mlp.gate_proj.weight"]
        new = server.dinov2_v5_state(old, expected)
        self.assertEqual(new[prefix + "attention.q_proj.weight"], "q")
        self.assertEqual(new[prefix + "attention.o_proj.bias"], "o")
        self.assertEqual(new[prefix + "mlp.gate_proj.weight"], [1, 2])
        self.assertEqual(new[prefix + "mlp.up_proj.weight"], [3, 4])
        self.assertEqual(new[prefix + "mlp.down_proj.weight"], "d")
        self.assertEqual(new[prefix + "norm1.weight"], "n")
        self.assertNotIn(prefix + "mlp.weights_in.weight", new)
        # a module that still uses the old names, or a checkpoint without DINOv2: untouched
        self.assertIs(server.dinov2_v5_state(old, list(old)), old)
        other = {"blocks.0.weight": 1}
        self.assertIs(server.dinov2_v5_state(other, expected), other)

    def load(self, extra):
        text = MINIMAL + f"""
[profiles.hy]
mesh_model = "tencent/Hunyuan3D-2mv"
gpus = ["RTX 4090"]
{extra}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kami.toml"
            path.write_text(text, encoding="utf-8")
            return config.load_config(path)

    def test_the_licence_must_be_accepted(self):
        with self.assertRaisesRegex(ConfigError, "European Union.*accept_hunyuan_license"):
            self.load("")
        cfg = self.load("accept_hunyuan_license = true")
        self.assertTrue(cfg.profiles["hy"].accept_hunyuan_license)

    def test_repos_are_the_folders_it_downloads(self):
        p = self.load("accept_hunyuan_license = true").profiles["hy"]
        self.assertEqual(p.mesh_backend, "hunyuan")
        self.assertEqual(p.repos, ["tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv",
                                   "tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo",
                                   "tencent/Hunyuan3D-2/hunyuan3d-delight-v2-0", "briaai/RMBG-2.0"])
        other = self.load('accept_hunyuan_license = true\nmesh_subfolder = "custom"').profiles["hy"]
        self.assertEqual(other.repos[0], "tencent/Hunyuan3D-2mv/custom")

    def test_download_counts_only_the_folder(self):
        files = (("hunyuan3d-dit-v2-mv/model.fp16.safetensors", 5), ("hunyuan3d-dit-v2-0/model.safetensors", 7),
                 ("README.md", 1))
        with mock.patch.object(fit, "repo_files", return_value=files):
            self.assertEqual(fit.entry_files("tencent/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv"),
                             [("hunyuan3d-dit-v2-mv/model.fp16.safetensors", 5)])
            self.assertEqual(len(fit.entry_files("tencent/Hunyuan3D-2mv")), 2)
        self.assertEqual(fit.split_repo("a/b/c/d"), ("a/b", "c/d"))

    def test_the_pod_gets_the_backend(self):
        cfg = self.load("accept_hunyuan_license = true")
        env = pod.server_spec(cfg, cfg.profiles["hy"], None, "key")[0]
        self.assertEqual((env["KAMI_MESH_BACKEND"], env["KAMI_MESH_SUBFOLDER"], env["KAMI_PAINT_MODEL"]),
                         ("hunyuan", "hunyuan3d-dit-v2-mv", "tencent/Hunyuan3D-2"))
        trellis = pod.server_spec(cfg, cfg.profiles["main"], None, "key")[0]
        self.assertNotIn("KAMI_MESH_BACKEND", trellis)


class PrintHintTests(unittest.TestCase):
    """Prompts get a hint so the picture is of something printable (jobs.PRINT_HINT)."""

    def test_the_hint_follows_the_prompt(self):
        sent = jobs.printable("a ceramic owl", jobs.PRINT_HINT)
        self.assertTrue(sent.startswith("a ceramic owl, stylised figurine"))
        for said in ("flat base", "one solid piece", "natural colours", "matte"):
            self.assertIn(said, sent)
        self.assertEqual(jobs.printable("owl", jobs.PRINT_HINT, 2), jobs.printable("owl", jobs.PRINT_HINT, 7))
        self.assertNotIn("{", sent)
        self.assertEqual(jobs.printable("owl", ""), "owl")
        self.assertEqual(jobs.printable("owl", "  "), "owl")
        self.assertEqual(jobs.printable("owl", "low poly {like} this, {colors} inks", 3),
                         "owl, low poly {like} this, 3 inks")        # only {colors} is filled in

    def args(self, **kw):
        return types.SimpleNamespace(**{"prompt": "an owl", "colors": None, "as_written": False, **kw})

    def test_the_cli_sends_it_unless_as_written(self):
        cfg = config.fallback_config()
        with mock.patch.object(cli.ui, "hint") as hint:
            self.assertEqual(cli.print_prompt(self.args(), cfg), jobs.printable("an owl", jobs.PRINT_HINT))
            self.assertIn("--as-written", hint.call_args[0][0])
            self.assertIn("natural colours", cli.print_prompt(self.args(colors=3), cfg))
        with mock.patch.object(cli.ui, "hint") as hint:
            self.assertEqual(cli.print_prompt(self.args(as_written=True), cfg), "an owl")
            hint.assert_not_called()
        with self.assertRaisesRegex(config.ConfigError, "--colors"):
            cli.print_prompt(self.args(colors=0), cfg)

    def test_kami_toml_replaces_it_or_turns_it_off(self):
        cfg = config.fallback_config()
        cfg.print_hint = ""
        with mock.patch.object(cli.ui, "hint") as hint:
            self.assertEqual(cli.print_prompt(self.args(), cfg), "an owl")
            hint.assert_not_called()
        cfg.print_hint = "{colors} inks"
        with mock.patch.object(cli.ui, "hint"):
            self.assertEqual(cli.print_prompt(self.args(colors=2), cfg), "an owl, 2 inks")
        self.assertEqual(config.DEFAULT_KEYS["print_hint"], config.TEXT)
        self.assertEqual(cli.WebBackend(cfg).print_hint, "{colors} inks")
        self.assertEqual(cli.WebBackend(config.fallback_config()).print_hint, jobs.PRINT_HINT)

    def test_the_pages_prompts_get_it_and_keep_their_names(self):
        hint = "flat base, {colors} colours"
        image, = webui.build_bodies({"kind": "image", "prompt": "an owl"}, hint)
        self.assertEqual(image[0], "Image: an owl")
        self.assertEqual((image[1]["prompt"], image[1]["name"]), ("an owl, flat base, 4 colours", "an-owl"))
        views, = webui.build_bodies({"kind": "views", "prompt": "an owl", "colors": 2}, hint)
        self.assertEqual(views[1]["prompt"], "an owl, flat base, 2 colours")
        t2m, = webui.build_bodies({"kind": "text-to-mesh", "prompt": "an owl", "print_hint": False}, hint)
        self.assertEqual(t2m[1]["prompt"], "an owl")
        self.assertEqual(webui.build_bodies({"kind": "image", "prompt": "an owl"})[0][1]["prompt"], "an owl")
        for colors in (0, 17, "4", True):
            with self.assertRaisesRegex(ValueError, "colors"):
                webui.build_bodies({"kind": "image", "prompt": "an owl", "colors": colors}, hint)

    def test_presets(self):
        self.assertEqual(jobs.PRINT_HINT, jobs.PRINT_PRESETS["figure"][2])
        self.assertEqual(jobs.print_hint(), jobs.PRINT_HINT)
        self.assertEqual(jobs.print_hint(None, "flexi"), jobs.PRINT_PRESETS["flexi"][2])
        self.assertEqual(jobs.print_hint("my own", "flexi"), "my own")       # written out: it wins
        self.assertEqual(jobs.print_hint("", "flexi"), "")
        for name, (label, about, text, front) in jobs.PRINT_PRESETS.items():
            self.assertTrue(label and about, name)
            self.assertEqual(jobs.preset_front(name), front)
            if front:
                self.assertIn("front view", front, name)             # the camera LoRA turns from a front
                self.assertNotRegex(front, r"\b(no|not|without|never)\b", name)
            self.assertIn("matte", text, name)                               # no light painted in
            self.assertNotRegex(text, r"\b(no|not|without|never)\b", name)    # said in the positive
            self.assertLess(len(jobs.printable("an owl", text, 4)), 800, name)
            # colours are not counted in a prompt (the full Z-Image painted blocks of them),
            # nor called uniform (it left the figure unpainted): kami print reduces them
            self.assertNotRegex(text, r"\{colors\}|uniform|separated areas", name)
        # a flexi comes whole and smooth, lying on the plate: the joints are cut afterwards,
        # so nothing in its words may draw them, nor a base
        flexi = jobs.PRINT_PRESETS["flexi"][2]
        for said in ("lying on its belly directly on the ground", "body in a straight line",
                     "head resting on the ground", "natural colours in flat matte paint"):
            self.assertIn(said, flexi)
        # nor parts it may not have (a snake got arms from "legs")
        self.assertNotRegex(flexi, r"joint|hinge|segment|articulat|base\b|standing|upright|print-in-place|legs?\b|limbs?\b")
        self.assertIn("simplified natural anatomy", flexi)
        # "flat" belongs to the paint and the light alone: said of the body, it drew a cut-out
        self.assertNotRegex(flexi.split("its natural colours")[0], r"\bflat\b|even thickness")
        # it is `figure`, which draws well, with only its stance changed: a longer hint drew itself
        figure = jobs.PRINT_PRESETS["figure"][2]
        self.assertTrue(flexi.startswith("stylised") and figure.startswith("stylised"))
        # the style simplifies the subject; it must not tell it what shape to be
        for text in (flexi, figure):
            self.assertNotRegex(text, r"toy|chunky|thick|rounded shapes|cute")
        self.assertLess(len(flexi), len(figure) + 70)
        self.assertIsNone(jobs.preset_front("flexi"))
        self.assertIsNone(jobs.preset_front("figure"))
        self.assertIsNone(jobs.preset_front(None))

    def test_presets_name_the_unwanted_only_in_the_negative_prompt(self):
        self.assertEqual(set(jobs.PRINT_NEGATIVES), set(jobs.PRINT_PRESETS))
        flexi = jobs.preset_negative("flexi")
        for unwanted in ("base", "joints", "separate segments", "glossy", "coiled"):
            self.assertIn(unwanted, flexi)
            self.assertNotIn(unwanted, jobs.PRINT_PRESETS["flexi"][2])     # never in the prompt itself
        self.assertNotIn("base", jobs.preset_negative("figure"))           # a figure stands on one
        self.assertIsNone(jobs.preset_negative(None))
        cfg = config.fallback_config()
        self.assertEqual(cli.with_negative({}, self.args(preset="flexi"), cfg), {"negative": flexi})
        self.assertEqual(cli.with_negative({"negative": "mine"}, self.args(preset="flexi"), cfg), {"negative": "mine"})
        self.assertEqual(cli.with_negative({}, self.args(as_written=True), cfg), {})
        cfg.print_hint = "my own words"                                    # no preset in force: nothing added
        self.assertEqual(cli.with_negative({"steps": 30}, self.args(), cfg), {"steps": 30})

    def test_the_page_sends_the_fronts_own_parameters_with_a_views_job(self):
        views, = webui.build_bodies({"kind": "views", "prompt": "an owl", "params": {
            "steps": 40, "image": {"negative": "base, joints", "steps": 32, "guidance": 4}}}, "")
        self.assertEqual(views[1]["params"], {"steps": 40, "image": {"negative": "base, joints", "steps": 32,
                                                                    "guidance": 4.0}})
        plain, = webui.build_bodies({"kind": "views", "prompt": "an owl", "params": {"image": {}}}, "")
        self.assertNotIn("image", plain[1]["params"])
        for bad in ({"image": {"lora_scale": 1}}, {"image": "x"}, "x"):
            with self.assertRaises(ValueError):
                webui.build_bodies({"kind": "views", "prompt": "an owl", "params": bad}, "")

    def test_the_shipped_profiles_make_fronts_with_models_that_obey(self):
        cfg = config.load_config(config.SOURCE_ROOT / "kami.toml")
        views, fronts = cfg.profiles["views"], cfg.profiles["fronts-flux"]
        # views turns the camera; its own front model is the quick one, for one-go jobs
        self.assertEqual(views.image_model, "Tongyi-MAI/Z-Image-Turbo")
        self.assertTrue(views.views_model and views.offload)
        # FLUX.2 for the fronts worth keeping: its own pod, it does not fit next to Qwen-Image-Edit
        self.assertEqual((fronts.image_model, fronts.views_model, fronts.mesh_model),
                         ("diffusers/FLUX.2-dev-bnb-4bit", "", ""))
        self.assertFalse(fronts.offload)

    def test_taking_something_away_is_called_out(self):
        self.assertEqual(jobs.negations("cute caterpillar without antenna, no legs, Not shiny"), ["without", "no", "Not"])
        self.assertEqual(jobs.negations("a knot of snow in a notebook"), [])
        self.assertEqual(jobs.negations(""), [])
        cfg = config.fallback_config()
        with mock.patch.object(cli.ui, "hint"), mock.patch.object(cli.ui, "warn") as warn:
            cli.print_prompt(self.args(prompt="a caterpillar without antennae"), cfg)
            self.assertIn("\"without\" in the prompt", warn.call_args[0][0])
            warn.reset_mock()
            cli.print_prompt(self.args(), cfg)
            warn.assert_not_called()

    def test_a_presets_front_goes_with_the_views_job(self):
        cfg = config.fallback_config()
        self.assertEqual(cli.preset_in_force(self.args(preset="flexi"), cfg), "flexi")
        self.assertEqual(cli.preset_in_force(self.args(), cfg), "figure")
        self.assertIsNone(cli.preset_in_force(self.args(preset="flexi", as_written=True), cfg))
        cfg.print_hint = "my own"
        self.assertIsNone(cli.preset_in_force(self.args(), cfg))

    def test_a_preset_is_picked_in_kami_toml_or_by_flag(self):
        cfg = config.fallback_config()
        with mock.patch.object(cli.ui, "hint") as hint:
            sent = cli.print_prompt(self.args(preset="flexi", colors=3), cfg)
            self.assertEqual(sent, jobs.printable("an owl", jobs.PRINT_PRESETS["flexi"][2], 3))
            self.assertIn("`flexi` printing preset", hint.call_args[0][0])
        cfg.print_preset = "bust"
        with mock.patch.object(cli.ui, "hint"):
            self.assertIn("sculpted bust", cli.print_prompt(self.args(), cfg))
            self.assertIn("low-poly", cli.print_prompt(self.args(preset="lowpoly"), cfg))
        self.assertEqual(cli.WebBackend(cfg).print_preset, "bust")
        cfg.print_hint = "my own words"
        with mock.patch.object(cli.ui, "hint") as hint:
            self.assertEqual(cli.print_prompt(self.args(), cfg), "an owl, my own words")
            self.assertIn("kami.toml's printing hint", hint.call_args[0][0])
            self.assertIn("lying on its belly", cli.print_prompt(self.args(preset="flexi"), cfg))    # the flag wins
        backend = cli.WebBackend(cfg)
        self.assertEqual((backend.print_hint, backend.print_preset), ("my own words", None))
        self.assertEqual(config.print_preset("flexi"), "flexi")
        with self.assertRaisesRegex(config.ConfigError, "not a preset"):
            config.print_preset("statue")

    def test_the_page_may_write_its_own_hints(self):
        hint = "flat base, {colors} colours"
        own, = webui.build_bodies({"kind": "image", "prompt": "an owl", "hint": " low poly, {colors} inks ",
                                   "colors": 2, "front_hint": "front view, white background"}, hint)
        self.assertEqual(own[1]["prompt"], "an owl, low poly, 2 inks, front view, white background")
        self.assertEqual(own[1]["name"], "an-owl")
        none, = webui.build_bodies({"kind": "image", "prompt": "an owl", "hint": "", "front_hint": ""}, hint)
        self.assertEqual(none[1]["prompt"], "an owl")                      # written empty: nothing added
        bare, = webui.build_bodies({"kind": "image", "prompt": "an owl", "print_hint": False,
                                    "front_hint": "front view"}, hint)
        self.assertEqual(bare[1]["prompt"], "an owl, front view")
        # a views job's front hint is the pod's to add: it goes as a parameter
        views, = webui.build_bodies({"kind": "views", "prompt": "an owl", "hint": "matte",
                                     "params": {"front_hint": "seen from the front", "elevation": "high-angle shot",
                                                "distance": "wide shot"}}, hint)
        self.assertEqual(views[1]["prompt"], "an owl, matte")
        self.assertEqual(views[1]["params"], {"front_hint": "seen from the front", "elevation": "high-angle shot",
                                              "distance": "wide shot"})
        for bad in ({"hint": 5}, {"hint": "x" * 2001}, {"front_hint": ["a"]}):
            with self.assertRaises(ValueError):
                webui.build_bodies({"kind": "image", "prompt": "an owl", **bad}, hint)

    def test_the_pod_takes_the_front_hint_as_a_parameter(self):
        from kami import server
        self.assertEqual(server.VIEWS_DEFAULTS["front_hint"], server.FRONT_HINT)
        self.assertEqual(server.front_prompt("an owl", server.VIEWS_DEFAULTS), f"an owl, {server.FRONT_HINT}")
        self.assertEqual(server.front_prompt("an owl", {"front_hint": "from the front"}), "an owl, from the front")
        self.assertEqual(server.front_prompt("an owl", {"front_hint": "  "}), "an owl")
        self.assertIn("front_hint", config.VIEWS_PARAMS)
        with mock.patch.object(server.PIPES, "image", object()):
            server.check_views_params({"front_hint": "x"}, has_front=False)
            with self.assertRaisesRegex(ValueError, "front_hint must be text"):
                server.check_views_params({"front_hint": 3}, has_front=False)


class MultiviewDefault(unittest.TestCase):
    def test_every_view_at_every_step_unless_asked_otherwise(self):
        from kami import server
        # the gecko's spots stayed sharp this way and blurred with one view a step
        self.assertEqual(server.MESH_DEFAULTS.get("multiview_mode"), "multidiffusion")
        self.assertEqual(set(server.MULTIVIEW_MODES), set(webui.MULTIVIEW_MODES))


class NetworkPatience(unittest.TestCase):
    """A job on the pod is not thrown away because this PC lost the network for a while."""

    def test_asking_again_while_the_pod_cannot_be_reached(self):
        answers = [jobs.Unreachable("Could not reach the pod: no DNS"), jobs.Unreachable("again"), "fine"]

        def call():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        told = []
        self.assertEqual(jobs.patiently(call, sleep=lambda s: None, waiting=told.append), "fine")
        self.assertEqual(len(told), 2)
        with self.assertRaisesRegex(jobs.Unreachable, "may still be on the pod"):
            jobs.patiently(mock.Mock(side_effect=jobs.Unreachable("Could not reach the pod: x")),
                           patience=0, sleep=lambda s: None)
        # a refusal is an answer: it is not asked again
        refused = mock.Mock(side_effect=jobs.JobError("Checking the job: 404"))
        with self.assertRaises(jobs.JobError):
            jobs.patiently(refused, sleep=lambda s: None)
        refused.assert_called_once()

    def test_a_request_that_gets_no_answer_is_unreachable(self):
        with mock.patch.object(jobs.urllib.request, "urlopen", side_effect=jobs.urllib.error.URLError("getaddrinfo failed")), \
                mock.patch.object(jobs.time, "sleep"):
            with self.assertRaisesRegex(jobs.Unreachable, "Could not reach the pod"):
                jobs._request("https://x/v1/jobs/j", "k")
        self.assertTrue(issubclass(jobs.Unreachable, jobs.JobError))

    def test_a_job_survives_a_gap_in_the_network(self):
        done = (200, json.dumps({"id": "j", "status": "done", "files": ["a.glb"]}).encode())
        gap = jobs.Unreachable("Could not reach the pod: getaddrinfo failed")
        with mock.patch.object(jobs, "_request", side_effect=[gap, gap, done]), mock.patch.object(jobs.ui, "Progress"):
            job = jobs.wait("https://x", "k", {"id": "j", "status": "running"}, "Mesh", sleep=lambda s: None)
        self.assertEqual(job["status"], "done")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(jobs, "_request", side_effect=[gap, (200, b"glb")]):
            saved = jobs.download("https://x", "k", job, Path(tmp), sleep=lambda s: None)
            self.assertEqual([p.read_bytes() for p in saved], [b"glb"])

    def test_the_page_keeps_asking_too(self):
        backend = mock.Mock()
        backend.target.return_value = ("https://pod", "key")
        gap = jobs.Unreachable("Could not reach the pod: getaddrinfo failed")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(jobs, "submit", return_value={"id": "r1", "status": "running"}), \
                mock.patch.object(jobs, "check", side_effect=[gap, {"id": "r1", "status": "done", "files": []}]), \
                mock.patch.object(jobs, "download", return_value=[]):
            work = webui.Work(backend, Path(tmp), sleep=lambda s: None)
            work.jobs["w1"] = {"id": "w1", "status": "sending"}
            work._run("w1", {"kind": "image"})
            self.assertEqual(work.jobs["w1"]["status"], "done")


class PageFixes(unittest.TestCase):
    def test_the_library_leaves_out_working_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("owl-front.png", "_sheet.png", ".hidden.png", "gemini/gecko-front.png",
                         "gemini/_mesh-comparison.png", "_sheets/owl-sheet.png", ".kami/x.png"):
                path = folder / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"x")
            self.assertEqual(sorted(p.relative_to(folder).as_posix() for p in webui.results(folder)),
                             ["gemini/gecko-front.png", "owl-front.png"])

    def test_the_pod_says_whether_its_image_model_reads_a_negative_prompt(self):
        from kami import server
        pipe = mock.Mock()
        pipe.__call__ = lambda prompt, negative_prompt=None, guidance_scale=0: None
        for model, guidance, image, expected in (
                ("Tongyi-MAI/Z-Image", 4.0, pipe, True),
                ("Tongyi-MAI/Z-Image-Turbo", 0.0, pipe, False),          # no guidance: ignored
                ("diffusers/FLUX.2-dev-bnb-4bit", 4.0, pipe, False),     # guidance-distilled
                ("Tongyi-MAI/Z-Image", 4.0, None, None),                 # not loaded yet
                ("", 4.0, pipe, False)):
            with mock.patch.object(server, "IMAGE_MODEL", model), \
                    mock.patch.dict(server.IMAGE_DEFAULTS, {"guidance": guidance}), \
                    mock.patch.object(server.PIPES, "image", image):
                self.assertEqual(server.reads_negative(), expected, model)
        info = {"pod": {"id": "p1", "url": "https://p1", "status": "RUNNING"}, "ready": True, "waiting": None}
        with mock.patch.object(cli, "load_state", return_value={"pod_id": "p1", "api_key": "k"}), \
                mock.patch.object(cli, "status_info", return_value=info), \
                mock.patch.object(jobs, "info", return_value={"image_model": "m", "image_negative": False}):
            served = cli.WebBackend(config.load_config()).status()["served"]
        self.assertIs(served["image_negative"], False)

