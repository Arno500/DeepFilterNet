"""Rewrite DeepFilterNet3 ONNX sub-models for OpenVINO / NPU.

Input: a DeepFilterNet*_onnx*.tar.gz bundle with enc.onnx, erb_dec.onnx,
df_dec.onnx, config.ini (the stock DeepFilterNet pulsed-ONNX export).

Transforms applied:

  1. Every ONNX `GRU` op already carries `initial_h` as input[5] and `Y_h` as
     output[1] (implicit, dropped by the stock graph). Promote each to a
     dedicated graph input/output tensor `h<N>_in` / `h<N>_out`, so the
     recurrent state can be threaded from the host at every call. This is
     required by the Intel NPU plugin, which rejects implicitly-stateful
     graphs.

  2. Pin every `S` (sequence) symbolic dim to 1 on graph-level I/O. Combined
     with the already-static batch=1 and feature dims, this yields a fully
     static graph the NPU plugin can compile ahead of time.

  3. Re-run ONNX shape inference so downstream consumers see concrete shapes.

Output: a parallel tarball (<stem>_npu.tar.gz) with patched ONNX files plus a
manifest JSON describing per-model hidden-state sockets.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
from onnx import TensorProto, helper, shape_inference


HIDDEN_SIZE = 512
HIDDEN_STATE_SHAPE = [1, 1, HIDDEN_SIZE]  # [num_directions * num_layers=1, batch=1, hidden]


def _set_dim_static(value_info: onnx.ValueInfoProto, dim_name_to_value: Dict[str, int]) -> None:
    shape = value_info.type.tensor_type.shape
    for d in shape.dim:
        if d.HasField("dim_param") and d.dim_param in dim_name_to_value:
            d.dim_value = dim_name_to_value[d.dim_param]
            d.ClearField("dim_param")


def _make_state_value_info(name: str, hidden_size: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, hidden_size])


def patch_model(model: onnx.ModelProto, model_name: str) -> Tuple[onnx.ModelProto, List[Dict]]:
    g = model.graph
    gru_nodes = [n for n in g.node if n.op_type == "GRU"]
    sockets: List[Dict] = []

    for idx, node in enumerate(gru_nodes):
        h_in = f"h{idx}_in"
        h_out = f"h{idx}_out"

        # GRU schema: X, W, R, B?, sequence_lens?, initial_h?, -> Y, Y_h
        while len(node.input) < 6:
            node.input.append("")
        node.input[5] = h_in

        while len(node.output) < 2:
            node.output.append("")
        node.output[1] = h_out

        # Add graph-level input/output.
        g.input.append(_make_state_value_info(h_in, HIDDEN_SIZE))
        g.output.append(_make_state_value_info(h_out, HIDDEN_SIZE))

        sockets.append(
            {
                "gru_node": node.name,
                "input": h_in,
                "output": h_out,
                "shape": HIDDEN_STATE_SHAPE,
                "dtype": "float32",
            }
        )

    # Pin S=1 on all pre-existing graph I/O.
    dim_map = {"S": 1}
    for vi in list(g.input) + list(g.output):
        _set_dim_static(vi, dim_map)

    # Re-run shape inference so consumers get concrete shapes.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        # Shape inference can throw on model fragments that still have
        # dim_param entries dangling in value_info; not fatal for runtime.
        print(f"[{model_name}] shape_inference warning: {e}", file=sys.stderr)

    return model, sockets


def _read_tarball(tar_path: Path) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            data = tf.extractfile(member).read()
            out[name] = data
    return out


def _write_tarball(tar_path: Path, entries: Dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name=f"export/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tar", type=Path)
    ap.add_argument("-o", "--output-tar", type=Path, default=None)
    ap.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="If set, also dump patched ONNX + manifest to this directory.",
    )
    args = ap.parse_args()

    if args.output_tar is None:
        stem = args.input_tar.name.replace(".tar.gz", "")
        args.output_tar = args.input_tar.with_name(f"{stem}_npu.tar.gz")

    entries = _read_tarball(args.input_tar)
    missing = {"enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"} - entries.keys()
    if missing:
        raise SystemExit(f"Input tarball missing files: {missing}")

    manifest: Dict[str, object] = {
        "source": str(args.input_tar),
        "hidden_size": HIDDEN_SIZE,
        "models": {},
    }

    patched: Dict[str, bytes] = {"config.ini": entries["config.ini"]}
    for name in ("enc.onnx", "erb_dec.onnx", "df_dec.onnx"):
        model = onnx.load_model_from_string(entries[name])
        patched_model, sockets = patch_model(model, name)
        onnx.checker.check_model(patched_model)
        patched[name] = patched_model.SerializeToString()
        manifest["models"][name.replace(".onnx", "")] = sockets
        print(f"[{name}] patched: {len(sockets)} GRU state sockets")

    patched["npu_manifest.json"] = json.dumps(manifest, indent=2).encode()

    _write_tarball(args.output_tar, patched)
    print(f"wrote {args.output_tar}")

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for name, data in patched.items():
            (args.dump_dir / name).write_bytes(data)
        print(f"dumped patched files to {args.dump_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
