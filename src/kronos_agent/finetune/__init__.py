"""Fine-tuning Kronos-small for BTC (step 6, SPEC section 9; ``docs/RETRAINING.md``).

Nothing is imported here on purpose: ``kronos.weights`` imports ``finetune.registry``, which stays
light, and the other modules of the package pull in numpy, pandas or torch.
``kronos_agent.runtime.configure_runtime()`` must run before anything imports torch.
"""
