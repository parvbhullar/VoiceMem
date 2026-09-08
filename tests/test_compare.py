"""A/B compare fan-out: two arms, one turn, memory injected vs not.

Arms are injected fake providers, so these tests do no network calls and load
no models. Run directly (this repo has no pytest in the demo venv)::

    python tests/test_compare.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web.compare import (Arm, CompareState, fan_out, parse_arms,  # noqa: E402
                         register_routes, sanitize)


def _provider(chunks, fail_after=None):
    """Fake reply provider. Records every (text, ctx) it was called with."""
    calls = []

    async def fn(text, memory_context=""):
        calls.append((text, memory_context))
        for i, chunk in enumerate(chunks):
            if fail_after is not None and i == fail_after:
                raise RuntimeError("provider blew up")
            yield chunk

    fn.calls = calls
    return fn


def _collect():
    """Fake ``send``: appends every ws message it was handed."""
    sent = []

    async def send(msg):
        sent.append(msg)

    return send, sent


def _run(arms, provider_map, text="what did I say?", ctx="- user has a cat", system="be nice"):
    send, sent = _collect()
    result = asyncio.run(fan_out(text, ctx, arms, send,
                                 system=system,
                                 provider=lambda arm: provider_map[arm.label]))
    return result, sent


class ContextRoutingTest(unittest.TestCase):
    def test_memory_arm_gets_context_and_plain_arm_gets_empty_string(self):
        a = _provider(["hi"])
        b = _provider(["hi"])
        arms = (Arm("a", model="m1", memory=True), Arm("b", model="m1", memory=False))

        _run(arms, {"a": a, "b": b}, ctx="- user has a cat")

        self.assertEqual(a.calls[0][1], "- user has a cat")
        self.assertEqual(b.calls[0][1], "")

    def test_both_arms_get_the_same_user_text(self):
        a, b = _provider(["x"]), _provider(["y"])
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        _run(arms, {"a": a, "b": b}, text="where do I work?")

        self.assertEqual(a.calls[0][0], "where do I work?")
        self.assertEqual(b.calls[0][0], "where do I work?")

    def test_empty_context_stays_empty_for_a_memory_arm(self):
        """No _NO_MEMORY_NOTE substitution: a memory arm with nothing retrieved
        must not be told 'you have no memories' — that is a third condition."""
        a = _provider(["x"])
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        _run(arms, {"a": a, "b": _provider(["y"])}, ctx="")

        self.assertEqual(a.calls[0][1], "")


class StreamingTest(unittest.TestCase):
    def test_deltas_are_labelled_and_replies_do_not_cross_contaminate(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        result, sent = _run(arms, {"a": _provider(["Hel", "lo"]),
                                   "b": _provider(["By", "e"])})

        self.assertEqual(result["a"], "Hello")
        self.assertEqual(result["b"], "Bye")
        deltas = [m for m in sent if m["type"] == "cmp_delta"]
        self.assertEqual("".join(m["text"] for m in deltas if m["panel"] == "a"), "Hello")
        self.assertEqual("".join(m["text"] for m in deltas if m["panel"] == "b"), "Bye")

    def test_start_and_done_are_sent_once_per_panel(self):
        arms = (Arm("a", model="gpt-4o", memory=True), Arm("b", model="gpt-4o", memory=False))

        _, sent = _run(arms, {"a": _provider(["x"]), "b": _provider(["y"])})

        starts = [m for m in sent if m["type"] == "cmp_start"]
        dones = [m for m in sent if m["type"] == "cmp_done"]
        self.assertEqual(sorted(m["panel"] for m in starts), ["a", "b"])
        self.assertEqual(sorted(m["panel"] for m in dones), ["a", "b"])
        self.assertEqual([m["memory"] for m in starts if m["panel"] == "a"], [True])
        self.assertEqual([m["memory"] for m in starts if m["panel"] == "b"], [False])


class TtfbTest(unittest.TestCase):
    """Time to first token, per arm.

    Total time is the wrong number to show in a voice demo: it grows with how
    long the answer is, so a chatty arm looks slower than a terse one even when
    it started speaking first. What the listener actually feels is the wait
    before the first word.
    """

    def test_first_delta_carries_the_ttfb(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        _, sent = _run(arms, {"a": _provider(["one", "two", "three"]),
                              "b": _provider(["x"])})

        deltas_a = [m for m in sent if m["type"] == "cmp_delta" and m["panel"] == "a"]
        self.assertIn("ttfb", deltas_a[0], "first delta must announce ttfb")
        self.assertNotIn("ttfb", deltas_a[1], "later deltas must not repeat it")
        self.assertNotIn("ttfb", deltas_a[2])

    def test_done_reports_both_ttfb_and_total(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        result, sent = _run(arms, {"a": _provider(["one", "two"]),
                                   "b": _provider(["y"])})

        done_a = next(m for m in sent if m["type"] == "cmp_done" and m["panel"] == "a")
        self.assertIn("ttfb", done_a)
        self.assertIn("ms", done_a)
        self.assertLessEqual(done_a["ttfb"], done_a["ms"])
        self.assertIn("a", result["ttfb_ms"])
        self.assertIn("b", result["ttfb_ms"])

    def test_an_arm_that_produced_nothing_reports_no_ttfb(self):
        """Never invent a number for a turn that never started."""
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        result, sent = _run(arms, {"a": _provider(["x"], fail_after=0),
                                   "b": _provider(["ok"])})

        done_a = next(m for m in sent if m["type"] == "cmp_done" and m["panel"] == "a")
        self.assertIsNone(done_a.get("ttfb"))
        self.assertIsNone(result["ttfb_ms"].get("a"))
        self.assertIsNotNone(result["ttfb_ms"]["b"])


class ArmFailureTest(unittest.TestCase):
    def test_one_arm_failing_does_not_stop_the_other(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        result, sent = _run(arms, {"a": _provider(["ok", " done"]),
                                   "b": _provider(["par", "tial"], fail_after=1)})

        self.assertEqual(result["a"], "ok done")
        self.assertIn("provider blew up", result["errors"]["b"])
        done_b = next(m for m in sent if m["type"] == "cmp_done" and m["panel"] == "b")
        self.assertIn("provider blew up", done_b["error"])
        done_a = next(m for m in sent if m["type"] == "cmp_done" and m["panel"] == "a")
        self.assertNotIn("error", done_a)

    def test_both_arms_failing_reports_a_turn_level_error(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))

        result, sent = _run(arms, {"a": _provider(["x"], fail_after=0),
                                   "b": _provider(["y"], fail_after=0)})

        self.assertEqual(result["a"], "")
        self.assertEqual(result["b"], "")
        self.assertEqual(len(result["errors"]), 2)
        self.assertTrue(any(m["type"] == "error" for m in sent))


class ClientGoneTest(unittest.TestCase):
    """A viewer closing the tab mid-turn is not an arm failure.

    Swallowing it made the server log two bogus panel errors and then raise
    RuntimeError('Cannot call "send" once a close message has been sent.') with
    a full traceback, because fan_out still tried to report the turn-level
    error on the dead socket.
    """

    def _send_that_dies_mid_stream(self, exc):
        """Succeeds for cmp_ctx / cmp_start, dies on the first delta.

        That is the shape of a real disconnect: the turn has already begun
        streaming when the viewer closes the tab. A send that fails on the
        very first message never reaches the code path that was broken.
        """
        sent = []

        async def send(msg):
            if msg["type"] == "cmp_delta":
                raise exc
            sent.append(msg)
        return send, sent

    def test_a_disconnect_mid_stream_propagates_instead_of_becoming_an_arm_error(self):
        from starlette.websockets import WebSocketDisconnect
        arms = (Arm("a", memory=True), Arm("b", memory=False))
        send, _ = self._send_that_dies_mid_stream(WebSocketDisconnect(1001))

        with self.assertRaises(WebSocketDisconnect):
            asyncio.run(fan_out("hi", "ctx", arms, send,
                                provider=lambda arm: _provider(["x", "y"])))

    def test_a_closed_socket_runtimeerror_mid_stream_propagates_too(self):
        arms = (Arm("a", memory=True), Arm("b", memory=False))
        exc = RuntimeError('Cannot call "send" once a close message has been sent.')
        send, _ = self._send_that_dies_mid_stream(exc)

        with self.assertRaises(RuntimeError):
            asyncio.run(fan_out("hi", "ctx", arms, send,
                                provider=lambda arm: _provider(["x", "y"])))

    def test_a_disconnect_does_not_report_a_turn_level_error_on_the_dead_socket(self):
        """The crash was the follow-up send, not the disconnect itself."""
        from starlette.websockets import WebSocketDisconnect
        arms = (Arm("a", memory=True), Arm("b", memory=False))
        send, sent = self._send_that_dies_mid_stream(WebSocketDisconnect(1001))

        with self.assertRaises(WebSocketDisconnect):
            asyncio.run(fan_out("hi", "ctx", arms, send,
                                provider=lambda arm: _provider(["x", "y"])))

        self.assertFalse([m for m in sent if m["type"] == "error"])
        self.assertFalse([m for m in sent if m["type"] == "cmp_done"])

    def test_an_unrelated_runtimeerror_from_send_is_not_treated_as_a_disconnect(self):
        """Only the closed-socket message counts; other RuntimeErrors are bugs
        we still want surfaced per panel rather than silently reclassified."""
        arms = (Arm("a", memory=True), Arm("b", memory=False))
        sent = []

        async def send(msg):
            sent.append(msg)
            if msg["type"] == "cmp_delta":
                raise RuntimeError("json encoding blew up")

        result = asyncio.run(fan_out("hi", "ctx", arms, send,
                                     provider=lambda arm: _provider(["x"])))

        self.assertEqual(len(result["errors"]), 2)
        self.assertIn("json encoding blew up", result["errors"]["a"])
        self.assertTrue(any(m["type"] == "error" for m in sent))


class SecretsTest(unittest.TestCase):
    def test_repr_does_not_leak_the_api_key(self):
        arm = Arm("a", model="gpt-4o", api_key="sk-supersecret")

        self.assertNotIn("sk-supersecret", repr(arm))
        self.assertEqual(arm.api_key, "sk-supersecret")

    def test_sanitize_reports_key_presence_without_the_key(self):
        state = CompareState(enabled=True,
                             arms=(Arm("a", model="gpt-4o", memory=True, api_key="sk-secret"),
                                   Arm("b", model="qwen", memory=False, base_url="http://x")))

        blob = sanitize(state)

        self.assertNotIn("sk-secret", repr(blob))
        self.assertTrue(blob["enabled"])
        self.assertEqual([a["label"] for a in blob["arms"]], ["a", "b"])
        self.assertTrue(blob["arms"][0]["has_key"])
        self.assertFalse(blob["arms"][1]["has_key"])
        self.assertEqual(blob["arms"][1]["base_url"], "http://x")
        for a in blob["arms"]:
            self.assertNotIn("api_key", a)


class ParseArmsTest(unittest.TestCase):
    def setUp(self):
        self.current = (Arm("a", model="gpt-4o", memory=True, api_key="sk-old"),
                        Arm("b", model="gpt-4o", memory=False))

    def test_absent_fields_keep_their_current_value(self):
        """The frontend posts only the switch the user flipped."""
        a, b = parse_arms([{"label": "b", "memory": True}], self.current)

        self.assertEqual(a.model, "gpt-4o")
        self.assertEqual(a.api_key, "sk-old")
        self.assertTrue(b.memory)
        self.assertEqual(b.model, "gpt-4o")

    def test_model_is_trimmed_and_replaces_the_old_one(self):
        a, _ = parse_arms([{"label": "a", "model": "  qwen2.5  "}], self.current)

        self.assertEqual(a.model, "qwen2.5")

    def test_empty_api_key_clears_the_stored_key(self):
        """That is how the user goes back to the server's own key."""
        a, _ = parse_arms([{"label": "a", "api_key": ""}], self.current)

        self.assertEqual(a.api_key, "")

    def test_unknown_panel_label_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_arms([{"label": "c", "model": "x"}], self.current)

    def test_non_string_model_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_arms([{"label": "a", "model": 7}], self.current)

    def test_empty_payload_is_a_no_op(self):
        self.assertEqual(parse_arms([], self.current), self.current)
        self.assertEqual(parse_arms(None, self.current), self.current)


class ApiTest(unittest.TestCase):
    """The two /api/compare routes, on a bare FastAPI app.

    register_routes() lives in compare.py rather than in build_app() so these
    run without importing web.utils, which pulls in torch and the TTS stack.
    """

    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self.state = CompareState()
        app = FastAPI()
        register_routes(app, lambda: self.state, self._set)
        self.client = TestClient(app)

    def _set(self, state):
        self.state = state

    def test_get_returns_the_default_state(self):
        """Compare is ON by default: the demo's whole point is the two replies
        side by side, so it must not need a click to appear."""
        body = self.client.get("/api/compare").json()

        self.assertTrue(body["enabled"])
        self.assertEqual([a["label"] for a in body["arms"]], ["a", "b"])
        self.assertTrue(body["arms"][0]["memory"])
        self.assertFalse(body["arms"][1]["memory"])

    def test_compare_can_be_switched_off(self):
        body = self.client.post("/api/compare", json={"enabled": False}).json()

        self.assertFalse(body["enabled"])
        self.assertFalse(self.state.enabled)

    def test_post_enables_compare_and_updates_one_arm(self):
        body = self.client.post("/api/compare", json={
            "enabled": True,
            "arms": [{"label": "b", "model": "qwen2.5", "memory": False}],
        }).json()

        self.assertTrue(body["enabled"])
        self.assertTrue(self.state.enabled)
        self.assertEqual(self.state.arms[1].model, "qwen2.5")
        self.assertEqual(self.state.arms[0].model, "")

    def test_post_never_echoes_the_api_key(self):
        r = self.client.post("/api/compare", json={
            "enabled": True,
            "arms": [{"label": "a", "api_key": "sk-do-not-echo"}],
        })

        self.assertNotIn("sk-do-not-echo", r.text)
        self.assertTrue(r.json()["arms"][0]["has_key"])
        self.assertEqual(self.state.arms[0].api_key, "sk-do-not-echo")

    def test_post_with_an_unknown_panel_is_a_400_and_keeps_the_old_state(self):
        self.client.post("/api/compare", json={"enabled": True, "arms": []})

        r = self.client.post("/api/compare", json={
            "arms": [{"label": "z", "model": "x"}]})

        self.assertEqual(r.status_code, 400)
        self.assertTrue(self.state.enabled)

    def test_post_without_enabled_keeps_the_current_toggle(self):
        self.client.post("/api/compare", json={"enabled": True, "arms": []})

        self.client.post("/api/compare", json={
            "arms": [{"label": "a", "model": "gpt-4o"}]})

        self.assertTrue(self.state.enabled)
        self.assertEqual(self.state.arms[0].model, "gpt-4o")


if __name__ == "__main__":
    unittest.main(verbosity=2)
