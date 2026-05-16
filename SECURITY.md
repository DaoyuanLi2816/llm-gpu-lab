# Security

This project intentionally keeps its attack surface small. There is no network
service shipped by the package itself — `serve-llamacpp` is a thin wrapper
around the official `llama-server` binary, which you run on `localhost` at your
own discretion.

## Reporting a vulnerability

Open a private security advisory on GitHub or email the maintainer listed in
the repository profile.  Please do **not** open a public issue for a
suspected vulnerability.

## What we treat as in-scope

- Code-execution issues in the package (e.g., unsafe YAML loading, unsafe
  pickle deserialization).
- Path traversal in any CLI command that writes files.
- Logging that leaks credentials.

## What is **out** of scope

- Vulnerabilities in third-party libraries — please report those upstream.
- Anything that requires the attacker to already have shell access to your
  machine.
- `llama.cpp`'s HTTP server: it is upstream's component; report there.

## Defaults

- The doctor command only reads system info. It never uploads anything.
- All CLI commands run locally. No telemetry is sent anywhere.
- YAML configs are loaded with `yaml.safe_load`.
- Checkpoints are saved as `safetensors` where practical to avoid arbitrary
  code execution from `torch.load(..., weights_only=False)`.
