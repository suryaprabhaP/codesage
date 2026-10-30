"""Agent loop tests with a scripted fake model, so they run without Ollama."""

import json
from pathlib import Path

import pytest

from codesage.agent import MAX_TOOL_ROUNDS, Agent


def tool_call(name: str, call_id: str = "call_1", **args) -> tuple[str, dict]:
    return "tool_call", {"id": call_id, "name": name, "arguments": json.dumps(args)}


class FakeLLM:
    """Replays a scripted list of replies, one per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        yield from self.replies.pop(0)


class BrokenLLM:
    def stream_chat(self, messages, tools=None):
        raise ConnectionError("server down")
        yield


class FakeUI:
    def __init__(self, approve: bool = True, command_answer: str = "yes"):
        self.approve = approve
        self.command_answer = command_answer
        self.commands_asked = []
        self.tool_lines = []
        self.warnings = []

    def stream_reply(self, events):
        answer, calls = "", []
        for kind, value in events:
            if kind == "text":
                answer += value
            elif kind == "tool_call":
                calls.append(value)
        return answer, calls

    def show_tool_call(self, call, result):
        self.tool_lines.append((call["name"], result))

    def confirm_change(self, change):
        return self.approve

    def confirm_command(self, command):
        self.commands_asked.append(command.command)
        return self.command_answer

    def warn(self, text):
        self.warnings.append(text)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("print('hello')\n")
    return tmp_path.resolve()


def test_plain_answer(project):
    agent = Agent(FakeLLM([("text", "Hi there")]), project, FakeUI())
    agent.ask("hello")
    assert agent.messages[-1] == {"role": "assistant", "content": "Hi there"}


def test_tool_result_goes_back_to_model(project):
    llm = FakeLLM([tool_call("read_file", path="main.py")], [("text", "It prints hello.")])
    agent = Agent(llm, project, FakeUI())
    agent.ask("what does main.py do?")

    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[2]["tool_calls"][0]["id"] == agent.messages[3]["tool_call_id"]
    assert "print('hello')" in agent.messages[3]["content"]
    assert llm.calls == 2


def test_approved_edit_changes_the_file(project):
    llm = FakeLLM(
        [tool_call("edit_file", path="main.py", old_text="hello", new_text="bye")],
        [("text", "Done.")],
    )
    Agent(llm, project, FakeUI(approve=True)).ask("say bye")
    assert (project / "main.py").read_text() == "print('bye')\n"


def test_rejected_edit_leaves_the_file(project):
    llm = FakeLLM(
        [tool_call("edit_file", path="main.py", old_text="hello", new_text="bye")],
        [("text", "OK, what would you like instead?")],
    )
    ui = FakeUI(approve=False)
    agent = Agent(llm, project, ui)
    agent.ask("say bye")
    assert (project / "main.py").read_text() == "print('hello')\n"
    assert "rejected" in agent.messages[3]["content"]


def test_stops_after_max_rounds(project):
    replies = [[tool_call("list_files")] for _ in range(MAX_TOOL_ROUNDS)]
    ui = FakeUI()
    Agent(FakeLLM(*replies), project, ui).ask("loop forever")
    assert len(ui.tool_lines) == MAX_TOOL_ROUNDS
    assert ui.warnings


def test_failed_turn_restores_history(project):
    agent = Agent(BrokenLLM(), project, FakeUI())
    with pytest.raises(ConnectionError):
        agent.ask("hello")
    assert [m["role"] for m in agent.messages] == ["system"]


def test_always_allow_skips_the_second_question(project):
    llm = FakeLLM(
        [tool_call("run_command", command="echo hi")],
        [tool_call("run_command", command="echo hi", call_id="call_2")],
        [("text", "Done.")],
    )
    ui = FakeUI(command_answer="always")
    agent = Agent(llm, project, ui)
    agent.ask("run it twice")
    assert ui.commands_asked == ["echo hi"]
    assert agent.messages[3]["content"].startswith("Exit code: 0")
    assert agent.messages[5]["content"].startswith("Exit code: 0")


def test_always_allow_survives_clear(project):
    agent = Agent(FakeLLM(), project, FakeUI())
    agent.allowed_commands.add("pytest")
    agent.reset()
    assert "pytest" in agent.allowed_commands


def test_command_answer_no(project):
    llm = FakeLLM([tool_call("run_command", command="echo hi")], [("text", "OK.")])
    agent = Agent(llm, project, FakeUI(command_answer="no"))
    agent.ask("run it")
    assert "did not allow" in agent.messages[3]["content"]


def test_reset_clears_conversation(project):
    agent = Agent(FakeLLM([("text", "Hi")]), project, FakeUI())
    agent.ask("hello")
    agent.reset()
    assert [m["role"] for m in agent.messages] == ["system"]
