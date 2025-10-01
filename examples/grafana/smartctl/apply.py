import subprocess
import tempfile
from pathlib import Path

from grafana_foundation_sdk.cog.encoder import JSONEncoder

from examples.grafana.smartctl.smartctl import dashboard

json = JSONEncoder(sort_keys=True, indent=2).encode(dashboard().build())

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
    tmp.write(json)

subprocess.run(["grr", "apply", tmp.name], check=True)  # noqa: S603, S607
Path(tmp.name).unlink()
