"""Local-first trace recording for agent runs.

We do not pull in the full OpenTelemetry SDK by default; instead, spans are
recorded as JSONL (OTLP-like) to a local file, which can later be exported.
This keeps the dependency footprint tiny while still giving observability:
which model/tool ran, for how long, with what outcome.
"""
from .recorder import TraceRecorder, get_recorder
