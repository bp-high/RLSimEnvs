import importlib.util
import sys
from pathlib import Path

# make oracle/solve.py importable as `oracle_solve`
_spec = importlib.util.spec_from_file_location("oracle_solve", Path(__file__).resolve().parents[1] / "oracle" / "solve.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["oracle_solve"] = _mod
