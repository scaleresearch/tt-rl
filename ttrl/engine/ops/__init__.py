"""Flat re-export of every autograd op."""

from ttrl.engine.ops.elementwise import add, div, mul, sub
from ttrl.engine.ops.functional import (
    apply_done_mask,
    entropy,
    gather_onehot,
    huber,
    log_softmax,
    masks,
    non_finite_check,
    one_hot_device,
    one_hot_gather,
    one_hot_host,
    purge_lane_caches,
)
from ttrl.engine.ops.matmul import matmul, purge_matmul_caches
from ttrl.engine.ops.reduce import (
    argmax_dim_detached,
    max_dim_detached,
    mean_all,
    mean_dim,
    sum_all,
    sum_dim,
)
from ttrl.engine.ops.select import clamp, maximum, minimum
from ttrl.engine.ops.shape import concat, reshape, split
from ttrl.engine.ops.unary import (
    abs_, exp, log, neg, relu, sigmoid, sqrt, square, tanh,
)

__all__ = [
    "add", "div", "mul", "sub",
    "abs_", "exp", "log", "neg", "relu", "sigmoid", "sqrt", "square", "tanh",
    "clamp", "maximum", "minimum",
    "matmul",
    "argmax_dim_detached", "max_dim_detached", "mean_all", "mean_dim",
    "sum_all", "sum_dim",
    "concat", "reshape", "split",
    "apply_done_mask", "entropy", "gather_onehot", "huber", "log_softmax",
    "masks", "non_finite_check", "one_hot_device", "one_hot_gather",
    "one_hot_host",
    "purge_lane_caches", "purge_matmul_caches",
]
