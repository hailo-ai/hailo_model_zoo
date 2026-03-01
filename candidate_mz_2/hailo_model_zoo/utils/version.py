#!/usr/bin/env python
def get_version(package_name):
    # See: https://packaging.python.org/guides/single-sourcing-package-version/ (Option 5)
    # We assume that the installed package is actually the same one we import. This assumption may
    # break in some edge cases e.g. if the user modifies sys.path manually.
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return "unknown"
