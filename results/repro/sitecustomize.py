"""Process-local compatibility shim for model-defined Transformers caches.

The tested Ouro and Huginn remote modules assign custom ``key_cache`` and
``value_cache`` containers. Transformers 4.54 exposes read-only properties with
those names on the base Cache class, preventing the model subclasses from being
constructed. Removing the base compatibility properties restores the interface
expected by the remote modules without modifying the installed package.
"""

from transformers.cache_utils import Cache


for attribute in ("key_cache", "value_cache"):
    if isinstance(getattr(Cache, attribute, None), property):
        delattr(Cache, attribute)
