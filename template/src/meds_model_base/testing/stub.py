"""Detecting an unimplemented model, so a fresh repository is honest rather than red.

A generated repository ships **no model** — ``model.py`` declares the hooks its command DAG calls and
raises ``NotImplementedError`` from each. The contract, the commands, the configs and the CLI are all real
and testable at that point; the model is not.

The conformance tests that *do* need a model (the end-to-end chain, the designed-signal learnability test)
therefore skip while the stub is in place, naming what to implement. They start running the moment the
stub marker is removed — which is the point: they are the specification your model has to satisfy, not
scaffolding you delete.

Mark a model as implemented by deleting ``is_stub`` from the class (or setting it to ``False``).
"""

from __future__ import annotations

from typing import Any

#: Attribute a rendered stub sets on itself; absent (or false) once a real model is written.
STUB_ATTRIBUTE = "is_stub"


def is_stub(obj: Any) -> bool:
    """Whether ``obj`` is (or is an instance of) an unimplemented model stub.

    Examples:
        >>> class Stub:
        ...     is_stub = True
        >>> class Real:
        ...     pass
        >>> is_stub(Stub), is_stub(Real)
        (True, False)
        >>> is_stub(Stub())
        True
    """
    return bool(getattr(obj, STUB_ATTRIBUTE, False))


def skip_if_stub(*objs: Any) -> None:
    """``pytest.skip`` when any of ``objs`` is still a stub, with an actionable message.

    Called by the rendered conformance tests. Importing pytest lazily keeps this module usable outside a
    test run.
    """
    import pytest

    stubs = [o for o in objs if is_stub(o)]
    if not stubs:
        return
    names = ", ".join(getattr(o, "__name__", type(o).__name__) for o in stubs)
    pytest.skip(
        f"{names} is still the generated stub. Implement it in src/<your_model>/model.py and remove the "
        "`is_stub` marker; this test is the contract your model has to satisfy."
    )
