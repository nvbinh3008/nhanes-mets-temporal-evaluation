"""Exercise an installed wheel and synthetic metric checks without study data."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

root = Path(__file__).resolve().parents[1]
scripts = Path(sysconfig.get_path("scripts"))
dist = importlib.metadata.distribution("nhanes-rule-hei-full")
assert dist.version == "2.1.3.2rc2", dist.version
installed = Path(dist.locate_file("")).resolve()
assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
assert not installed.is_relative_to(root / "src"), installed
sys.dont_write_bytecode = True
sys.path.insert(0, str(root / "scripts"))
for helper in sorted((root / "scripts").glob("phaseb_*.py")):
    importlib.import_module(helper.stem)
print("PASS research helper imports without study inputs", flush=True)
import yaml
for name in ("pipeline", "hei_pipeline", "modeling"):
    with (root / "config" / f"{name}.yaml").open(encoding="utf-8") as stream:
        assert isinstance(yaml.safe_load(stream), dict)
print("PASS parsed three YAML configurations", flush=True)
for name in ("nhanes-etl", "hei-pipeline", "nhanes-full", "nhanes-model"):
    executable = scripts / (name + (".exe" if sys.platform == "win32" else ""))
    subprocess.run([str(executable), "--help"], check=True, cwd=root)
    print("PASS entry point:", name, flush=True)
executable = scripts / ("nhanes-full.exe" if sys.platform == "win32" else "nhanes-full")
result = subprocess.run([str(executable), "preflight"], check=True, cwd=root, capture_output=True, text=True)
checks = json.loads(result.stdout)
assert all(checks[k] for k in ("etl_config_exists", "hei_config_exists"))
assert not checks["dictionary_exists"], "RC2 requires a clean checkout without the controlled workbook."
assert not checks["ready_for_etl"]
assert not checks["ready_for_hei"], "Run smoke checks in an empty study checkout."
print("PASS configuration preflight; controlled workbook absent; HEI data readiness:", checks["ready_for_hei"], flush=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True, cwd=root)
print("PASS installed wheel and synthetic tests. No study re-execution.", flush=True)
