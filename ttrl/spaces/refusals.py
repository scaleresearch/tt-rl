"""Why an `Observation` is not an array, said at the point of the mistake.

Every way a reader might still treat an observation as the single array it
used to be is refused here, and each refusal names the leaf-wise form to
write instead. The refusals are the migration documentation this project
ships, so they live together rather than scattered through the class that
raises them. `ttrl.spaces.observation` binds them onto `Observation`.
"""

from __future__ import annotations

import numpy as np

#: Package-private: `ttrl.spaces.observation` imports these and is the
#: only module that raises them.
__all__ = ["_ARITH_OPS", "_BASICS", "_NDARRAY_ATTRS", "_arith", "_refuse",
           "_refuse_attr"]


#: What every refusal restates: for a reader mid-migration the message is
#: the whole documentation this project ships.
_FACT = ("an observation is a group of named leaf arrays now, and the single "
         "array it used to be is one leaf of that group")

_BASICS = ("Write obs[name] for one leaf's array, obs.leaves for the leaf "
           "names, and obs.lanes for the lane count (None for a single "
           "observation). A promoted single array sits under the one leaf "
           "ttrl.DEFAULT_LEAF, so obs[ttrl.DEFAULT_LEAF] is the array the "
           "old observation was.")

#: The ndarray attribute surface, read off numpy rather than typed out, so a
#: name numpy adds later is refused too. Private names are left out because
#: numpy probes `__array_struct__` on its way into `__array__`, where a
#: refusal would report the probe instead of the caller's conversion.
_NDARRAY_ATTRS = frozenset(n for n in dir(np.ndarray)
                           if not n.startswith("_"))

#: Source form per operator dunder, so a refusal quotes what was typed.
_ARITH_OPS = {
    "__add__": "obs + x", "__radd__": "x + obs", "__sub__": "obs - x",
    "__rsub__": "x - obs", "__mul__": "obs * x", "__rmul__": "x * obs",
    "__truediv__": "obs / x", "__rtruediv__": "x / obs",
    "__floordiv__": "obs // x", "__rfloordiv__": "x // obs",
    "__mod__": "obs % x", "__rmod__": "x % obs", "__pow__": "obs ** x",
    "__rpow__": "x ** obs", "__matmul__": "obs @ x", "__rmatmul__": "x @ obs",
    "__and__": "obs & x", "__rand__": "x & obs", "__or__": "obs | x",
    "__ror__": "x | obs", "__xor__": "obs ^ x", "__rxor__": "x ^ obs",
    "__lshift__": "obs << x", "__rlshift__": "x << obs",
    "__rshift__": "obs >> x", "__rrshift__": "x >> obs",
    "__divmod__": "divmod(obs, x)", "__rdivmod__": "divmod(x, obs)",
    "__neg__": "-obs", "__pos__": "+obs", "__invert__": "~obs",
    "__abs__": "abs(obs)", "__float__": "float(obs)", "__int__": "int(obs)",
    "__complex__": "complex(obs)", "__index__": "operator.index(obs)",
    "__lt__": "obs < x", "__le__": "obs <= x", "__gt__": "obs > x",
    "__ge__": "obs >= x",
}


def _refuse(did: str, instead: str):
    raise TypeError(f"{did}: {_FACT}. {instead}")


def _refuse_attr(name: str):
    if callable(getattr(np.ndarray, name, None)):
        _refuse(
            f"obs.{name}(...) calls the ndarray method {name!r} on an "
            f"Observation",
            f"Call it on the leaf you mean, as in obs[name].{name}(...), or "
            f"over every leaf with {{n: obs[n].{name}(...) for n in "
            f"obs.leaves}}. {_BASICS}",
        )
    _refuse(
        f"obs.{name} reads the ndarray attribute {name!r} off an Observation",
        f"A {name} belongs to one array: write obs[name].{name}. {_BASICS}",
    )


def _arith(name: str):
    def method(self, *args):
        _refuse(
            f"`{_ARITH_OPS[name]}` operates arithmetically on a whole "
            f"Observation",
            f"Operate on the leaf you mean, as in obs[name] in place of obs, "
            f"or over every leaf with {{n: obs[n] for n in obs.leaves}}. "
            f"{_BASICS}",
        )
    method.__name__ = name
    return method
