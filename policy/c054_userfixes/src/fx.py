"""Two backends for one controller source: numpy (float32, for simulation) and ONNX (graph export).

The controller is written once against `F` (a backend) using Python operators on column tensors of
shape [B, 1]. With `NP` the operators are plain float32 numpy; with `OnnxBuilder` every operator
emits one standard ONNX node, so the exported graph computes the same thing op for op.

Allowed in controller code: + - * / unary-, comparisons (< <= > >=), & | ~ on booleans, and the
F.* functions below. No Python `if`/`and`/`or` on tensors.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------- numpy backend
class NP:
    name = "numpy"

    @staticmethod
    def col(obs, i):
        return obs[:, i:i + 1]

    @staticmethod
    def const_like(x, v):
        return np.full_like(x, np.float32(v), dtype=np.float32)

    @staticmethod
    def where(c, a, b):
        return np.where(c, np.float32(a) if np.isscalar(a) else a, np.float32(b) if np.isscalar(b) else b).astype(np.float32)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b).astype(np.float32)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b).astype(np.float32)

    @staticmethod
    def clip(x, lo, hi):
        return np.minimum(np.maximum(x, np.float32(lo)), np.float32(hi)).astype(np.float32)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def atan(x):
        return np.arctan(x)

    @staticmethod
    def sin(x):
        return np.sin(x)

    @staticmethod
    def cos(x):
        return np.cos(x)

    @staticmethod
    def floor(x):
        return np.floor(x)

    @staticmethod
    def round(x):  # half to even, like ONNX Round
        return np.round(x)

    @staticmethod
    def tofloat(b):
        return b.astype(np.float32)

    @staticmethod
    def concat(cols):
        return np.concatenate(cols, axis=1).astype(np.float32)

    @staticmethod
    def linear(x, W):
        return np.matmul(x, np.asarray(W, np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------- ONNX backend
class Sym:
    __array_priority__ = 1000

    def __init__(self, b: "OnnxBuilder", name: str, kind: str = "f"):
        self.b, self.name, self.kind = b, name, kind

    def _bin(self, op, other, kind="f", rev=False):
        o = self.b.lift(other)
        x, y = (o, self) if rev else (self, o)
        return self.b.node(op, [x.name, y.name], kind)

    def __add__(self, o): return self._bin("Add", o)
    def __radd__(self, o): return self._bin("Add", o, rev=True)
    def __sub__(self, o): return self._bin("Sub", o)
    def __rsub__(self, o): return self._bin("Sub", o, rev=True)
    def __mul__(self, o): return self._bin("Mul", o)
    def __rmul__(self, o): return self._bin("Mul", o, rev=True)
    def __truediv__(self, o): return self._bin("Div", o)
    def __rtruediv__(self, o): return self._bin("Div", o, rev=True)
    def __neg__(self): return self.b.node("Neg", [self.name])
    def __lt__(self, o): return self._bin("Less", o, "b")
    def __le__(self, o): return self._bin("LessOrEqual", o, "b")
    def __gt__(self, o): return self._bin("Greater", o, "b")
    def __ge__(self, o): return self._bin("GreaterOrEqual", o, "b")
    def __and__(self, o): return self._bin("And", o, "b")
    def __rand__(self, o): return self._bin("And", o, "b", rev=True)
    def __or__(self, o): return self._bin("Or", o, "b")
    def __ror__(self, o): return self._bin("Or", o, "b", rev=True)
    def __invert__(self): return self.b.node("Not", [self.name], "b")
    def __bool__(self):
        raise TypeError("no Python control flow on tensors")


class OnnxBuilder:
    name = "onnx"

    def __init__(self):
        from onnx import helper, numpy_helper  # noqa: F401
        self.nodes, self.inits, self._consts, self._n = [], [], {}, 0

    def _fresh(self, p="t"):
        self._n += 1
        return f"{p}{self._n}"

    def node(self, op, inputs, kind="f", **attrs):
        from onnx import helper
        out = self._fresh()
        self.nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return Sym(self, out, kind)

    def constant(self, v, dtype=np.float32):
        from onnx import numpy_helper
        arr = np.asarray(v, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self._consts:
            nm = self._fresh("c")
            self.inits.append(numpy_helper.from_array(arr, nm))
            self._consts[key] = Sym(self, nm, "b" if dtype == np.bool_ else "f")
        return self._consts[key]

    def lift(self, x):
        if isinstance(x, Sym):
            return x
        if isinstance(x, (bool, np.bool_)):
            return self.constant(bool(x), np.bool_)
        return self.constant(float(x))

    # F interface -----------------------------------------------------------------
    def col(self, obs: Sym, i: int):
        idx = self.constant(np.array([i], np.int64), np.int64)
        return self.node("Gather", [obs.name, idx.name], axis=1)

    def const_like(self, x, v):
        return x * 0.0 + float(v)

    def where(self, c, a, b):
        a, b = self.lift(a), self.lift(b)
        return self.node("Where", [c.name, a.name, b.name])

    def minimum(self, a, b):
        return self.node("Min", [self.lift(a).name, self.lift(b).name])

    def maximum(self, a, b):
        return self.node("Max", [self.lift(a).name, self.lift(b).name])

    def clip(self, x, lo, hi):
        return self.minimum(self.maximum(x, lo), hi)

    def abs(self, x):
        return self.node("Abs", [x.name])

    def sqrt(self, x):
        return self.node("Sqrt", [x.name])

    def atan(self, x):
        return self.node("Atan", [x.name])

    def sin(self, x):
        return self.node("Sin", [x.name])

    def cos(self, x):
        return self.node("Cos", [x.name])

    def floor(self, x):
        return self.node("Floor", [x.name])

    def round(self, x):
        return self.node("Round", [x.name])

    def tofloat(self, b):
        from onnx import TensorProto
        return self.node("Cast", [b.name], to=TensorProto.FLOAT)

    def concat(self, cols):
        return self.node("Concat", [c.name for c in cols], axis=1)

    def linear(self, x, W):
        w = self.constant(np.asarray(W, np.float32))
        return self.node("MatMul", [x.name, w.name])

    def build(self, policy_fn, opset=17, name="policy"):
        from onnx import TensorProto, helper
        import onnx
        obs = Sym(self, "obs")
        out = policy_fn(self, obs)
        # rename the final output to "action"
        self.nodes.append(helper.make_node("Identity", [out.name], ["action"]))
        g = helper.make_graph(
            self.nodes, name,
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", 26])],
            [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 5])],
            self.inits,
        )
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="nist_ctrl")
        m.ir_version = 8
        onnx.checker.check_model(m)
        return m
