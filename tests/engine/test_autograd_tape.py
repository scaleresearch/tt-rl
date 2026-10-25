"""Host-only tests for the autograd tape's pure-Python logic."""

import pytest
import ttnn

pytest.importorskip("ttnn")  # module import needs the ttnn python package

from ttrl.engine.autograd import Tensor

from tests._fakes import TapeData as _FakeData


def _node(shape=(1, 1), parents=(), op="", requires_grad=True):
    return Tensor(_FakeData(shape), requires_grad, parents, op)


def _tape(root, order):
    """Wire every non-leaf node to record its visit order and push a gradient to its parents."""

    def install(node):
        for parent in node._parents:
            install(parent)
        if not node._op:
            return

        def make(n):
            def _backward(grad):
                order.append(n._op)
                for parent in n._parents:
                    if parent.grad is None:
                        parent.grad = grad
            return _backward

        node._backward = make(node)

    install(root)


def _seedless_backward(root):
    """`backward()` seeds `self.grad` with `ttnn.full_like`, which needs a real device tensor."""

    real_full_like, real_typecast = ttnn.full_like, ttnn.typecast
    ttnn.full_like = lambda data, value: _FakeData(data.shape)
    ttnn.typecast = lambda t, dtype: t
    try:
        root.backward()
    finally:
        ttnn.full_like, ttnn.typecast = real_full_like, real_typecast


def test_backward_visits_children_before_parents_with_shared_subgraph():
    """The diamond."""
    leaf = _node(op="")
    d = _node(parents=(leaf,), op="d")
    b = _node(parents=(d,), op="b")
    c = _node(parents=(d,), op="c")
    a = _node(parents=(b, c), op="a")

    order = []
    _tape(a, order)
    _seedless_backward(a)

    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("d") > order.index("b")
    assert order.index("d") > order.index("c")


def test_backward_tears_down_the_tape_but_spares_leaves():
    leaf = _node(op="")
    mid = _node(parents=(leaf,), op="mid")
    root = _node(parents=(mid,), op="root")

    _tape(root, [])
    _seedless_backward(root)

    assert mid._backward_done is True
    assert root._backward_done is True
    # A parameter is reused as the start of the next step's graph forever.
    assert leaf._backward_done is False
    assert mid._parents == ()
    assert root._parents == ()


def test_backward_refuses_a_second_pass():
    leaf = _node(op="")
    root = _node(parents=(leaf,), op="root")
    _tape(root, [])
    _seedless_backward(root)

    with pytest.raises(RuntimeError, match="second time"):
        _seedless_backward(root)


def test_backward_rejects_non_scalar():
    root = _node(shape=(2, 3), parents=(_node(op=""),), op="root")
    _tape(root, [])
    with pytest.raises(RuntimeError, match="non-scalar"):
        _seedless_backward(root)


def test_backward_rejects_a_tensor_with_no_tape():
    """A node built under no_grad (or a detached one) has no gradient to give."""
    detached = _node(op="mul", requires_grad=False)
    with pytest.raises(RuntimeError, match="requires no gradient"):
        _seedless_backward(detached)


