"""Framework adapters: each wraps one model family behind a core component contract.

Frameworks are imported only inside `_load`, so these modules are importable
(and their options validated) without the GPU stack.
"""
