import unittest

from web.session_context import SessionBuffer


def test_committed_turn_is_removed_from_session_context():
    buffer = SessionBuffer()
    committed = buffer.add("s1", "demo", "I have an interview tomorrow", "Let's prepare")
    pending = buffer.add("s1", "demo", "What should I wear?", "A dark suit")

    buffer.mark_complete(committed, committed=True)

    rendered = buffer.render("s1", "demo")
    assert "interview tomorrow" not in rendered
    assert "What should I wear?" in rendered
    assert [turn.turn_id for turn in buffer.turns("s1", "demo")] == [pending]


def test_uncommitted_turn_remains_and_spaces_are_isolated():
    buffer = SessionBuffer(text_limit=20)
    turn_id = buffer.add("s1", "zh", "a quick side topic", "okay, go on")
    buffer.add("s1", "en", "temporary topic", "go on")
    buffer.add("s2", "zh", "another session", "should not leak in")

    buffer.mark_complete(turn_id, committed=False)

    assert "a quick side topic" in buffer.render("s1", "zh")
    assert "another session" not in buffer.render("s1", "zh")
    assert "temporary topic" in buffer.render("s1", "en")
    assert "not yet stored" in buffer.render("s1", "en", language="en")

    buffer.clear_session("s1")
    assert not buffer.turns("s1", "zh")
    assert buffer.turns("s2", "zh")


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        unittest.FunctionTestCase(test_committed_turn_is_removed_from_session_context),
        unittest.FunctionTestCase(test_uncommitted_turn_remains_and_spaces_are_isolated),
    ])
