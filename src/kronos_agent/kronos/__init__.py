"""Kronos integration: vendored code import, verified weights, forecasting.

Nothing is imported here on purpose: ``kronos_agent.runtime.configure_runtime()`` must run
before any of these modules pulls in ``torch`` or ``huggingface_hub``.
"""
