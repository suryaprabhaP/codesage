from pathlib import Path

import pytest

from codesage.tools import run_tool


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A small fake project to run the tools against."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("def greet(name):\n    return f'Hello {name}'\n")
    (tmp_path / "README.md").write_text("# Demo\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("greet = None\n")
    return tmp_path.resolve()


def test_list_files_skips_ignored_folders(project):
    result = run_tool(project, "list_files", "{}")
    assert result.splitlines() == ["README.md", "app/main.py"]


def test_read_file_adds_line_numbers(project):
    result = run_tool(project, "read_file", '{"path": "app/main.py"}')
    assert result.splitlines()[0] == "   1  def greet(name):"


def test_read_file_line_range(project):
    result = run_tool(project, "read_file", '{"path": "app/main.py", "start_line": 2}')
    assert result == "   2      return f'Hello {name}'"


def test_search_code_finds_matches(project):
    result = run_tool(project, "search_code", '{"pattern": "GREET"}')
    assert result == "app/main.py:1: def greet(name):"


def test_search_code_accepts_invalid_regex(project):
    result = run_tool(project, "search_code", '{"pattern": "greet("}')
    assert "app/main.py:1" in result


def test_path_outside_project_is_refused(project):
    result = run_tool(project, "read_file", '{"path": "../secret.txt"}')
    assert result.startswith("Error:") and "outside the project" in result


def test_missing_file_returns_error(project):
    assert run_tool(project, "read_file", '{"path": "nope.py"}').startswith("Error:")


def test_bad_json_and_unknown_tool_return_errors(project):
    assert run_tool(project, "read_file", "{not json").startswith("Error:")
    assert run_tool(project, "delete_everything", "{}").startswith("Error:")


def test_dict_arguments_are_accepted(project):
    assert "Demo" in run_tool(project, "read_file", {"path": "README.md"})
