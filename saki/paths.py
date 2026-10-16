"""Resolve private historical locations through explicit user configuration."""
import os
import re

def release_path(template):
    def substitute(match):
        key = 'SAKI_' + match.group(1)
        value = os.environ.get(key)
        if not value:
            raise RuntimeError(f'Set {key} before using this source-run adapter; see docs/REPRODUCTION.md')
        return value
    return re.sub(r'\$\{([A-Z_]+)\}', substitute, template)
