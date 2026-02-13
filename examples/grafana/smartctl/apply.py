import subprocess
import tempfile
from pathlib import Path

from grafana_foundation_sdk.cog.encoder import JSONEncoder

from examples.grafana.smartctl.smartctl import manifest

json = JSONEncoder(sort_keys=True, indent=2).encode(manifest())

with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".json", delete=False) as tmp:
  tmp.write(json)

cmd = ["grafanactl", "resources", "push", "dashboards", "--path", tmp.name]
subprocess.run(cmd, check=True)  # noqa: S603
Path(tmp.name).unlink()
