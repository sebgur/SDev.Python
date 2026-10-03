import ast
from pathlib import Path
import pytest
import sdevpy

SDEVPY_DIR = Path(sdevpy.__file__).parent


def _sdevpy_imports(file: Path):
    """ Absolute module names imported by file """
    for node in ast.walk(ast.parse(file.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == 'sdevpy':  # from sdevpy import datapaths
                for alias in node.names:
                    yield f"sdevpy.{alias.name}"
            else:
                yield node.module


@pytest.mark.parametrize('package, allowed', [
    ('market', {'market', 'conventions', 'utilities', 'maths', 'datapaths'}),
    ('conventions', {'conventions', 'utilities'}),
])
def test_package_dependencies(package, allowed):
    violations = []
    for file in (SDEVPY_DIR / package).rglob('*.py'):
        for module in _sdevpy_imports(file):
            parts = module.split('.')
            if parts[0] == 'sdevpy' and len(parts) > 1 and parts[1] not in allowed:
                violations.append(f"{file.relative_to(SDEVPY_DIR)}: {module}")
    assert not violations, "\n".join(violations)
