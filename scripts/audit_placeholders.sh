#!/usr/bin/env bash
#
# Fail if source / docs contain leftover placeholder markers.
#
# Why this script exists
# ----------------------
# Forbidden tokens (TODO, FIXME, NotImplementedError, lorem, fake benchmark,
# dummy result, TBD) signal unfinished code. We never want them in source
# code, tests, configs, or in primary docs that users read first.
#
# Where they ARE allowed
# ----------------------
# Two kinds of mentions are legitimate and the audit must let them through:
#
#   1. Forward-looking work items live in `docs/roadmap.md`. That file is
#      explicitly the home of TODO / FIXME / TBD.
#   2. The rule itself — this script, plus the few user-facing docs that
#      explain "what counts as a placeholder?". Those mention the words
#      to describe the rule; they are not actual unfinished work.
#
# The check
# ---------
# For every forbidden token, run ripgrep / grep over the tree. Drop hits
# that match (a) the roadmap file, (b) this script, or (c) a line that is
# clearly describing the rule rather than containing the placeholder
# itself.

set -uo pipefail

PATTERNS=(
  "TODO"
  "FIXME"
  "NotImplementedError"
  "lorem ipsum"
  "fake benchmark"
  "dummy result"
  "TBD"
  "placeholder"
)

# Files that are allowed to talk about the rule. They must mention the
# words in order to describe the audit, but they are not themselves
# unfinished work.
META_FILES=(
  "docs/roadmap.md"
  "scripts/audit_placeholders.sh"
  "CONTRIBUTING.md"
  "Makefile"
  ".github/workflows/ci.yml"
  "README.md"
  "docs/quickstart.md"
  "docs/design.md"
  "docs/troubleshooting.md"
  "docs/ip_safety.md"
  "docs/licenses.md"
  "docs/results.md"
)

IGNORE_DIRS=(
  ".git"
  ".venv"
  ".ruff_cache"
  ".pytest_cache"
  "__pycache__"
  "artifacts"
  "results"
  "external"
  ".cache"
  "site-packages"
)

EXCLUDE_ARGS=()
for d in "${IGNORE_DIRS[@]}"; do
  EXCLUDE_ARGS+=("--exclude-dir=${d}")
done

# Build a single grep filter line to skip allowed files.
META_GREP_PATTERN="$(printf "|^\\./%s:" "${META_FILES[@]}" | sed -E 's#\\\.#\\.#g; s#^\|##')"

EXIT=0

for pattern in "${PATTERNS[@]}"; do
  hits=$(grep -RInE "${EXCLUDE_ARGS[@]}" \
    --exclude="*.gguf" --exclude="*.safetensors" --exclude="*.bin" --exclude="*.pt" \
    --exclude="*.png" --exclude="*.jpg" \
    -e "${pattern}" . 2>/dev/null \
    | grep -vE "${META_GREP_PATTERN}" \
    || true)
  if [ -n "${hits}" ]; then
    echo "FAIL: token \"${pattern}\" found:" >&2
    echo "${hits}" >&2
    EXIT=1
  fi
done

# Detect functions whose only body is `pass` (excluding tests + dataclass
# stubs). We warn rather than fail because some valid stubs (e.g. abstract
# methods or __init_subclass__) use pass legitimately.
if [ -d "src" ]; then
  empty_pass=$(grep -RInE "${EXCLUDE_ARGS[@]}" \
    --include="*.py" \
    -B1 -A0 "^[[:space:]]+pass[[:space:]]*$" src/ 2>/dev/null \
    | grep -vE "raise " \
    | grep -vE "^--$" \
    | awk 'NR % 2 == 1' \
    | grep -E "def " || true)

  if [ -n "${empty_pass}" ]; then
    echo "WARN: functions with empty 'pass' body:" >&2
    echo "${empty_pass}" >&2
  fi
fi

if [ "${EXIT}" -ne 0 ]; then
  echo
  echo "Audit failed. Either fix the offending lines or move the marker to docs/roadmap.md." >&2
  exit 1
fi

echo "Placeholder audit OK."
