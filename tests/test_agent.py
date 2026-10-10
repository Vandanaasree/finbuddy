"""Offline tests: no API key needed. A scripted fake LLM stands in for the real model,
so these tests verify the graph logic (tool loop, human approval, database) itself.

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import finbuddy.db as db
from finbuddy.graph import build_graph
from main import run_turn


class ScriptedLLM:
    """Returns pre-written AI messages in order. Mimics the two methods the graph uses."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class FinBuddyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db.DB_PATH = os.path.join(self.tmp.name, "test.db")
        self.cfg = {"configurable": {"thread_id": "t1"}}

    def tearDown(self):
        self.tmp.cleanup()

    def app(self, replies):
        return build_graph(ScriptedLLM(replies), "tester", MemorySaver())

    def rows(self):
        with db.connect() as c:
            return c.execute("SELECT * FROM transactions").fetchall()

    def test_tool_loop_saves_expense(self):
        app = self.app(
            [
                call("add_transaction", {"amount": 450, "category": "food", "description": "lunch"}, "c1"),
                AIMessage(content="Logged 450 for lunch."),
            ]
        )
        reply = run_turn(app, self.cfg, "I spent 450 on lunch")
        self.assertEqual(reply, "Logged 450 for lunch.")
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["amount"], rows[0]["category"]), (450, "food"))

    def test_large_transaction_needs_approval_and_can_be_rejected(self):
        app = self.app(
            [
                call("add_transaction", {"amount": 50000, "category": "shopping"}, "c1"),
                AIMessage(content="Okay, I did not save it."),
            ]
        )
        # Auto-answer "no" by monkeypatching input
        import builtins

        original = builtins.input
        builtins.input = lambda _="": "n"
        try:
            reply = run_turn(app, self.cfg, "I bought a laptop for 50000")
        finally:
            builtins.input = original
        self.assertEqual(reply, "Okay, I did not save it.")
        self.assertEqual(self.rows(), [])  # rejected -> nothing saved

    def test_large_transaction_approved(self):
        app = self.app(
            [
                call("add_transaction", {"amount": 50000, "category": "shopping"}, "c1"),
                AIMessage(content="Saved."),
            ]
        )
        run_turn(app, self.cfg, "I bought a laptop for 50000", auto_approve=True)
        self.assertEqual(len(self.rows()), 1)

    def test_delete_requires_approval(self):
        with db.connect() as c:
            c.execute(
                "INSERT INTO transactions (user_id, date, kind, amount, category) "
                "VALUES ('tester', '2026-10-01', 'expense', 100, 'food')"
            )
        app = self.app([call("delete_transaction", {"transaction_id": 1}, "c1"), AIMessage(content="Deleted.")])
        import builtins

        original = builtins.input
        builtins.input = lambda _="": "n"
        try:
            run_turn(app, self.cfg, "delete transaction 1")
        finally:
            builtins.input = original
        self.assertEqual(len(self.rows()), 1)  # still there

        app2 = self.app([call("delete_transaction", {"transaction_id": 1}, "c2"), AIMessage(content="Deleted.")])
        run_turn(app2, {"configurable": {"thread_id": "t2"}}, "delete transaction 1", auto_approve=True)
        self.assertEqual(self.rows(), [])

    def test_summary_and_budget_tools(self):
        from finbuddy.tools import make_tools

        tools = {t.name: t for t in make_tools("tester")}
        tools["add_transaction"].invoke({"amount": 300, "category": "Food", "date_str": "2026-10-02"})
        tools["add_transaction"].invoke({"amount": 700, "category": "food", "date_str": "2026-10-03"})
        tools["add_transaction"].invoke({"amount": 200, "category": "transport", "date_str": "2026-10-03"})
        out = tools["spending_summary"].invoke({"start_date": "2026-10-01", "end_date": "2026-10-31"})
        self.assertIn("food: 1000.00", out)
        self.assertIn("TOTAL expense: 1200.00", out)
        tools["set_budget"].invoke({"category": "food", "monthly_limit": 900})
        status = tools["budget_status"].invoke({"month": "2026-10"})
        self.assertIn("OVER BUDGET", status)

    def test_users_are_isolated(self):
        from finbuddy.tools import make_tools

        a = {t.name: t for t in make_tools("alice")}
        b = {t.name: t for t in make_tools("bob")}
        a["add_transaction"].invoke({"amount": 10, "category": "food"})
        self.assertEqual(b["list_transactions"].invoke({}), "No matching transactions.")

    def test_bad_input_is_handled(self):
        from finbuddy.tools import make_tools

        t = {x.name: x for x in make_tools("tester")}
        self.assertIn("Error", t["add_transaction"].invoke({"amount": -5, "category": "food"}))
        self.assertIn("Error", t["add_transaction"].invoke({"amount": 5, "category": "food", "date_str": "yesterday"}))
        self.assertIn("Unsupported", t["convert_currency"].invoke({"amount": 1, "from_currency": "XYZ", "to_currency": "INR"}))


if __name__ == "__main__":
    unittest.main()
