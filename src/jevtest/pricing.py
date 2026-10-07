"""Frozen provider price constants shared by adapters and report runners."""

# S1 probe: Jev charges $42 per billion input tokens; output tokens are free.
# Source and capture date: docs/jev-s1-probe.md.
JEV_INPUT_USD_PER_TOKEN = 42.0 / 1_000_000_000
