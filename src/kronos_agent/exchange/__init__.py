"""Exchange clients.

Two independent modules with no shared code: ``public_market`` (real-market data, public GET,
no keys, pinned to ``open-api.bingx.com``) and ``demo_account`` (authenticated, pinned to
``open-api-vst.bingx.com``, the only module allowed to contain signing code).
"""
