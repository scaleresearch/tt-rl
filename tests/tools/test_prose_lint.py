"""Every comment and docstring that names a repo path or dotted name names one that exists."""

from pathlib import Path

from tools.lint_prose import ROOTS, violations

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_comment_or_docstring_names_code_that_is_not_there():
    for tree in ROOTS:
        assert sorted((REPO_ROOT / tree).rglob("*.py")), f"{tree} holds no modules"
    vs = violations(REPO_ROOT)
    assert not vs, "prose names code that is not there:\n" + "\n".join(str(v) for v in vs)
