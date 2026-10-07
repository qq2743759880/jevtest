import shutil
import subprocess
import sys


def _help_output() -> str:
    executable = shutil.which("jevtest")
    command = [executable] if executable else [sys.executable, "-m", "jevtest"]
    proc = subprocess.run([*command, "--help"], capture_output=True, text=True, check=True)
    return proc.stdout


def test_help_lists_subcommands():
    output = _help_output()
    assert "run" in output
    assert "benchmark" in output
    assert "synthgen" in output
