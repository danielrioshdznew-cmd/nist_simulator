"""4e: export refport_fx (env switches RF_* apply) to ONNX opset 17, dynamic batch, 184 inputs; checks ORT == numpy."""
import sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "ctrl")]
import numpy as np, onnx, onnxruntime as ort
from onnx import TensorProto, helper
import importlib, os, fx
R = importlib.import_module(os.environ.get('RF_MODULE', 'refport_fx'))
out = sys.argv[1]
b = fx.OnnxBuilder(); obs = fx.Sym(b, "obs"); act = R.policy(b, obs)
b.nodes.append(helper.make_node("Identity", [act.name], ["action"]))
g = helper.make_graph(b.nodes, "refport_fx", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", R.OBS_DIM])],
                      [helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 5])], b.inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="nist_refport_fx"); m.ir_version = 8
onnx.checker.check_model(m, full_check=True); onnx.save(m, out)
o = np.random.default_rng(0).normal(0, 0.3, (64, R.OBS_DIM)).astype(np.float32)
for k in range(8):
    o[:, 21 + 17 * k] = (np.random.default_rng(k).uniform(size=64) > 0.3)
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
d = float(np.abs(s.run(None, {"obs": o})[0] - np.asarray(R.policy(fx.NP, o))).max())
print("nodes", len(m.graph.node), "bytes", Path(out).stat().st_size, "sha256", hashlib.sha256(Path(out).read_bytes()).hexdigest()[:16], "max|ort-np|", d)
