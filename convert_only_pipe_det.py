#!/usr/bin/env python3
"""
Convert wireandpipe_detection_cpp/src/only_pipe_det.pt to ONNX.

Usage:
    python3 convert_only_pipe_det.py

The generated ONNX is placed next to the source PT:
    wireandpipe_detection_cpp/src/only_pipe_det.onnx

Export settings follow the existing wire_pipe_det.onnx conversion:
    - imgsz=640
    - opset=12
    - simplify=True
    - FP32 (half=False)
    - static batch=1 (dynamic=False)
"""

import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: {e}\nInstall with:  pip install ultralytics")
    sys.exit(1)


def convert(pt_path: Path, onnx_path: Path, imgsz: int = 640) -> Path:
    pt_path = pt_path.resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")

    onnx_path = onnx_path.resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"\nExport config:")
    print(f"  Input size : {imgsz}x{imgsz}")
    print(f"  Precision  : FP32 (half=False)")
    print(f"  Simplify   : True (ultralytics built-in)")
    print(f"  Opset      : 12")
    print(f"  Dynamic    : False (static batch=1)")
    print(f"  Output     : {onnx_path}\n")

    model.export(
        format='onnx',
        imgsz=imgsz,
        half=False,
        int8=False,
        simplify=True,
        opset=12,
        dynamic=False,
    )

    # Ultralytics generates the ONNX next to the PT by default.
    default_onnx = pt_path.with_suffix('.onnx')

    # Some versions append a size suffix.
    if not default_onnx.exists():
        candidates = sorted(pt_path.parent.glob(f"{pt_path.stem}*.onnx"))
        if candidates:
            default_onnx = candidates[0]

    if not default_onnx.exists():
        raise RuntimeError("Export failed: ONNX file not generated.")

    if default_onnx.resolve() != onnx_path.resolve():
        shutil.move(str(default_onnx), str(onnx_path))
        print(f"Moved to: {onnx_path}")
    else:
        print(f"Saved to: {onnx_path}")

    # Validate and print I/O information.
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print("\nONNX validation passed.")

        print("\nModel I/O:")
        for inp in m.graph.input:
            shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            print(f"  Input : {inp.name}  -> {shape}")
        for out in m.graph.output:
            shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            print(f"  Output: {out.name}  -> {shape}")
    except ImportError:
        print("\nSkip validation (onnx package not installed).")

    return onnx_path


def main() -> int:
    package_dir = Path(__file__).resolve().parent
    pt_path = package_dir / "src" / "only_pipe_det.pt"
    onnx_path = package_dir / "src" / "only_pipe_det.onnx"

    try:
        convert(pt_path, onnx_path, imgsz=640)
        print(f"\nDone. To use it in the C++ node, set:")
        print(f'  yolo_model_path: "{onnx_path}"')
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
