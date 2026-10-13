"""Telemetry installation for the harness (design §12, M12).

``setup()`` owns the OTel TracerProvider; ``ci_lab.obs`` (OTel API only) emits harness spans
and MAF emits GenAI spans through it. Exporters: OTLP/HTTP to a local Aspire dashboard
(``ci-lab dashboard up``) and JSONL span records under the run dir.
"""
