# ruff: noqa: S602
import json
import subprocess
import time
from collections.abc import Callable
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import psutil

K6_VUS = 250
K6_DURATION = "30s"


@dataclass
class Endpoint:
  name: str
  path: str
  method: str
  expected_response: dict
  request_body: dict | None = None


@dataclass
class ServerConfig:
  name: str
  language: str
  build_cmd: list[str] | None
  run_cmd: list[str]
  cwd: Path
  base_url: str
  endpoints: list[Endpoint]


@dataclass
class BenchmarkResult:
  server_name: str
  endpoint_name: str
  language: str
  timestamp: str
  measurements: list[float]
  mean_rps: float
  std_rps: float
  mean_latency_us: float


def build_server(server: ServerConfig) -> None:
  if server.build_cmd is None:
    return

  subprocess.run(server.build_cmd, shell=True, cwd=server.cwd, timeout=300, check=True)


def start_server(server: ServerConfig) -> subprocess.Popen:
  return subprocess.Popen(
    server.run_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    shell=True,
    cwd=server.cwd,
  )


def stop_server(process: subprocess.Popen) -> None:
  parent = psutil.Process(process.pid)
  children = parent.children(recursive=True)
  children.append(parent)

  for p in children:
    p.terminate()

  psutil.wait_procs(children, timeout=5)


def wait_for_server(server: ServerConfig, timeout: int = 30) -> None:
  endpoint = server.endpoints[0]
  start_time = time.time()

  while time.time() - start_time < timeout:
    try:
      with httpx.Client(timeout=1) as client:
        response = client.get(f"{server.base_url}{endpoint.path}")
        if response.status_code in {200, 405}:
          return
    except httpx.ConnectTimeout:
      pass

    time.sleep(0.5)

  msg = f"server {server.name} did not start within {timeout}s"
  raise TimeoutError(msg)


def verify_server(server: ServerConfig) -> None:
  for endpoint in server.endpoints:
    if not verify_endpoint(server, endpoint):
      msg = f"endpoint {endpoint.path} failed verification"
      raise RuntimeError(msg)


def verify_endpoint(server: ServerConfig, endpoint: Endpoint) -> bool:
  url = f"{server.base_url}{endpoint.path}"

  with httpx.Client(timeout=5) as client:
    match endpoint.method:
      case "GET":
        response = client.get(url)
      case "POST":
        response = client.post(url, json=endpoint.request_body)
      case "PUT":
        response = client.put(url, json=endpoint.request_body)
      case "DELETE":
        response = client.delete(url)
      case _:
        msg = f"unsupported method: {endpoint.method}"
        raise ValueError(msg)

    if response.status_code != 200:
      msg = f"endpoint {endpoint.path} returned {response.status_code}"
      raise RuntimeError(msg)

    if response.json() != endpoint.expected_response:
      msg = f"endpoint {endpoint.path} returned unexpected response"
      raise RuntimeError(msg)

    return True


def run_benchmark(server: ServerConfig, endpoint: Endpoint, script_dir: Path) -> dict:
  cmd = [
    "k6",
    "run",
    "--env",
    f"URL={server.base_url}{endpoint.path}",
    "--env",
    f"METHOD={endpoint.method}",
    "--env",
    f"VUS={K6_VUS}",
    "--env",
    f"DURATION={K6_DURATION}",
    str(script_dir / "k-six.ts"),
  ]

  if endpoint.request_body:
    cmd.insert(7, "--env")
    cmd.insert(8, f"BODY={json.dumps(endpoint.request_body)}")

  subprocess.run(cmd, shell=True, timeout=120, check=True)

  with (script_dir / ".build" / "summary.json").open() as f:
    return json.loads(f.read())


def save_result(result: BenchmarkResult, results_file: Path) -> None:
  data = {
    "server_name": result.server_name,
    "endpoint_name": result.endpoint_name,
    "language": result.language,
    "timestamp": result.timestamp,
    "measurements": result.measurements,
    "mean_rps": result.mean_rps,
    "std_rps": result.std_rps,
    "mean_latency_us": result.mean_latency_us,
  }

  with results_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps(data) + "\n")


def remove_outliers(values: list, k: float = 1.5) -> list:
  arr = np.array(values)

  Q1, Q3 = np.percentile(arr, [25, 75])
  IQR = Q3 - Q1

  lower = Q1 - k * IQR
  upper = Q3 + k * IQR
  mask = (arr >= lower) & (arr <= upper)

  return arr[mask].tolist()


@contextmanager
def status() -> Generator[Callable[[str], None]]:
  prev_len = 0

  def update(text: str) -> None:
    nonlocal prev_len

    padding = " " * max(0, prev_len - len(text))
    print(f"{text}{padding}", end="\r", flush=True)
    prev_len = len(text)

  try:
    yield update
  finally:
    print(" " * prev_len, end="\r", flush=True)
