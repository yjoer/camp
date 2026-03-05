import argparse
import datetime as dt
from pathlib import Path

import numpy as np

from examples.http._bench.utils import BenchmarkResult
from examples.http._bench.utils import Endpoint
from examples.http._bench.utils import ServerConfig
from examples.http._bench.utils import build_server
from examples.http._bench.utils import remove_outliers
from examples.http._bench.utils import run_benchmark
from examples.http._bench.utils import save_result
from examples.http._bench.utils import start_server
from examples.http._bench.utils import status
from examples.http._bench.utils import stop_server
from examples.http._bench.utils import verify_server
from examples.http._bench.utils import wait_for_server

script_dir = Path().cwd()
http_dir = script_dir.parent
react_server_dir = http_dir.parent / "react-server"

RESULTS_FILE = script_dir / ".build" / "results.jsonl"
WARMUP_ROUNDS = 1
TARGET_MEASUREMENTS = 10
IQR_MULTIPLIER = 1.25
MAX_STD_RATIO = 0.05

servers: list[ServerConfig] = [
  ServerConfig(
    name="node",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "server.ts"],
    cwd=http_dir / "node",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="express",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "server.ts"],
    cwd=http_dir / "express",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="fastify",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "server.ts"],
    cwd=http_dir / "fastify",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="h-three",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "server.ts"],
    cwd=http_dir / "h-three",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="trpc",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "node.ts"],
    cwd=http_dir / "trpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/trpc/hello", method="GET", expected_json={"result": {"data": {"hello": "world"}}}),
    ],
  ),
  ServerConfig(
    name="trpc-fastify",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "fastify.ts"],
    cwd=http_dir / "trpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/trpc/hello", method="GET", expected_json={"result": {"data": {"hello": "world"}}}),
    ],
  ),
  ServerConfig(
    name="orpc",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "node.ts"],
    cwd=http_dir / "orpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/hello", method="GET", expected_json={"json": {"hello": "world"}}),
    ],
  ),
  ServerConfig(
    name="orpc-fastify",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "fastify.ts"],
    cwd=http_dir / "orpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/rpc/hello", method="GET", expected_json={"json": {"hello": "world"}}),
    ],
  ),
  ServerConfig(
    name="orpc-h-three",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "h-three.ts"],
    cwd=http_dir / "orpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/rpc/hello", method="GET", expected_json={"json": {"hello": "world"}}),
    ],
  ),
  ServerConfig(
    name="orpc-openapi-node",
    language="typescript",
    build_cmd=None,
    run_cmd=["node", "openapi-node.ts"],
    cwd=http_dir / "orpc",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="fastapi",
    language="python",
    build_cmd=None,
    run_cmd=["uv", "run", "python", "server.py"],
    cwd=http_dir / "fastapi",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="spring-boot",
    language="java",
    build_cmd=["gradle", "bootJar"],
    run_cmd=[
      "java",
      "-XX:ActiveProcessorCount=1",
      "-jar",
      ".build/libs/spring-boot.jar",
    ],
    cwd=http_dir / "spring-boot",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="axum",
    language="rust",
    build_cmd=["cargo", "build", "--release"],
    run_cmd=["cargo", "run", "--release"],
    cwd=http_dir / "axum",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="axum-connect",
    language="rust",
    build_cmd=["cargo", "build", "--release"],
    run_cmd=["cargo", "run", "--release"],
    cwd=http_dir / "axum-connect",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/hello.v1.HelloService/Hello", method="POST", request_body={}, expected_json={"hello": "world"}),
    ],
  ),
  ServerConfig(
    name="react-server-vite",
    language="typescript",
    build_cmd=["pnpm", "build"],
    run_cmd=["pnpm", "start"],
    cwd=react_server_dir / "vite",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_regex="Hello, World!"),
    ],
  ),
  ServerConfig(
    name="react-server-vite-stream",
    language="typescript",
    build_cmd=["pnpm", "build"],
    run_cmd=["pnpm", "start:stream"],
    cwd=react_server_dir / "vite",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_regex="Hello, World!"),
    ],
  ),
  ServerConfig(
    name="react-server-next-pages",
    language="typescript",
    build_cmd=["pnpm", "build"],
    run_cmd=["pnpm", "start"],
    cwd=react_server_dir / "next",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_regex="Hello, World!"),
    ],
  ),
  ServerConfig(
    name="react-server-next-app",
    language="typescript",
    build_cmd=["pnpm", "build"],
    run_cmd=["pnpm", "start"],
    cwd=react_server_dir / "next",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/app", method="GET", expected_regex="Hello, World!"),
    ],
  ),
  ServerConfig(
    name="react-server-tanstack-start",
    language="typescript",
    build_cmd=["pnpm", "build"],
    run_cmd=["pnpm", "start"],
    cwd=react_server_dir / "tanstack-start",
    base_url="http://127.0.0.1:3000",
    endpoints=[
      Endpoint(name="hello", path="/", method="GET", expected_regex="Hello, World!"),
    ],
  ),
]

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default=None)
parser.add_argument("--verify-only", action="store_true")
args = parser.parse_args()

with status() as update:
  for server in servers:
    if args.server and not server.name.startswith(args.server): continue
    update(f"building {server.name}")
    build_server(server)

with status() as update:
  for server in servers:
    if args.server and not server.name.startswith(args.server): continue
    update(f"starting {server.name}")
    process = start_server(server)

    try:
      update(f"waiting for {server.name}")
      wait_for_server(server)

      update(f"verifying {server.name}")
      verify_server(server)
      if args.verify_only: continue

      for endpoint in server.endpoints:
        update(f"warming up {server.name}/{endpoint.name}")
        for _ in range(WARMUP_ROUNDS): run_benchmark(server, endpoint, script_dir)

        update(f"benchmarking {server.name}/{endpoint.name}")
        measurements: list[float] = []

        while True:
          summary = run_benchmark(server, endpoint, script_dir)
          rps = float(summary["metrics"]["http_reqs"]["values"]["rate"])
          measurements.append(rps)

          m = remove_outliers(measurements, k=IQR_MULTIPLIER)
          mean_rps = np.mean(m)
          std_rps = np.std(m)
          std_ratio = std_rps / mean_rps
          update(
            f"benchmarking {server.name}/{endpoint.name}, "
            f"count: {len(measurements)}, "
            f"outliers: {len(measurements) - len(m)}, "
            f"mean: {mean_rps}, "
            f"std: {std_rps}, "
            f"std_ratio: {std_ratio}",
          )

          if len(m) >= TARGET_MEASUREMENTS and std_ratio <= MAX_STD_RATIO: break

        m = remove_outliers(measurements, k=IQR_MULTIPLIER)
        mean_rps = np.mean(m)
        std_rps = np.std(m)
        mean_latency_us = 1_000_000 / mean_rps

        result = BenchmarkResult(
          server_name=server.name,
          endpoint_name=endpoint.name,
          language=server.language,
          timestamp=dt.datetime.now(tz=dt.UTC).isoformat(),
          measurements=measurements,
          mean_rps=mean_rps,
          std_rps=std_rps,
          mean_latency_us=mean_latency_us,
        )

        save_result(result, RESULTS_FILE)
    finally:
      update(f"stopping {server.name}")
      stop_server(process)
