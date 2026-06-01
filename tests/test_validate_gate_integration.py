import pytest
import os
import sys

def test_run_validate_gate():
    # Let's run scripts/validate_gate.py and see if it works under pytest!
    import subprocess
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(workspace_root, "scripts", "validate_gate.py")
    
    # Run with .venv/bin/python
    python_bin = os.path.join(workspace_root, ".venv", "bin", "python")
    
    res = subprocess.run([python_bin, script_path], capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    assert res.returncode == 0
