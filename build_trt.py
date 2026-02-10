"""Build TensorRT engines from ONNX models with dynamic batch support"""

import argparse
from pathlib import Path

import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    max_batch_size: int = 8,
    min_batch_size: int = 1,
    fp16: bool = True,
) -> None:
    """Build TensorRT engine from ONNX model with dynamic batch support"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Enable FP16 if supported
    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled")

    # Configure optimized batch size
    profile = builder.create_optimization_profile()

    # Get input tensors and set profiles
    for i in range(network.num_inputs):
        input_name = network.get_input(i).name
        input_shape = network.get_input(i).shape
        print(f"Input: {input_name}, shape: {input_shape}")

        # Check if this has multiple dynamic dimensions (e.g., input_boxes)
        num_dynamic = sum(1 for d in list(input_shape) if d == -1)

        min_shape = list(input_shape)
        opt_shape = list(input_shape)
        max_shape = list(input_shape)

        if num_dynamic == 1:
            # Only batch is dynamic
            min_shape[0] = min_batch_size
            opt_shape[0] = opt_batch_size
            max_shape[0] = max_batch_size
        elif num_dynamic > 1 and input_name == "input_boxes":
            # For input_boxes: dynamic batch and num_boxes
            # Set fixed num_boxes ranges
            min_num_boxes = 1
            opt_num_boxes = 5
            max_num_boxes = 10

            min_shape[0] = min_batch_size
            min_shape[1] = min_num_boxes

            opt_shape[0] = opt_batch_size
            opt_shape[1] = opt_num_boxes

            max_shape[0] = max_batch_size
            max_shape[1] = max_num_boxes
        elif num_dynamic > 1 and input_name == "input_boxes_labels":
            # For input_boxes_labels: dynamic batch and num_boxes
            # Set fixed num_boxes ranges matching input_boxes
            min_num_boxes = 1
            opt_num_boxes = 5
            max_num_boxes = 10

            min_shape[0] = min_batch_size
            min_shape[1] = min_num_boxes

            opt_shape[0] = opt_batch_size
            opt_shape[1] = opt_num_boxes

            max_shape[0] = max_batch_size
            max_shape[1] = max_num_boxes
        else:
            # For other inputs, only set batch
            dynamic_idx = next(
                (i for i, d in enumerate(list(input_shape)) if d == -1), None
            )
            if dynamic_idx is not None:
                min_shape[dynamic_idx] = min_batch_size
                opt_shape[dynamic_idx] = opt_batch_size
                max_shape[dynamic_idx] = max_batch_size

        profile.set_shape(
            input_name, tuple(min_shape), tuple(opt_shape), tuple(max_shape)
        )
        config.add_optimization_profile(profile)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)  # 4GB

    print("Building TensorRT engine...")
    print(
        f"  Min batch: {min_batch_size}, Opt batch: {opt_batch_size}, Max batch: {max_batch_size}"
    )

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build engine")
        return

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Engine saved to: {engine_path}")

    # Print engine info
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized_engine)
    print(f"Engine has {engine.num_io_tensors} I/O tensors")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"  {name} [{mode}]: {shape}, dtype: {dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from ONNX models"
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default="onnx-models-v2",
        help="Directory containing ONNX models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tensorrt-models-v2",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--opt-batch-size", type=int, default=1, help="Optimal batch size for profile"
    )
    parser.add_argument(
        "--max-batch-size", type=int, default=8, help="Maximum batch size for profile"
    )
    parser.add_argument(
        "--min-batch-size", type=int, default=1, help="Minimum batch size for profile"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use FP16 precision"
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=["vision", "text", "decoder"],
        default=None,
        help="Build specific module only (default: build all)",
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = []
    if args.module:
        modules = [args.module]
    else:
        if (onnx_dir / "vision-encoder.onnx").exists():
            modules.append("vision")
        if (onnx_dir / "text-encoder.onnx").exists():
            modules.append("text")
        if (onnx_dir / "decoder.onnx").exists():
            modules.append("decoder")

    for module in modules:
        if module == "vision":
            onnx_file = onnx_dir / "vision-encoder.onnx"
            engine_file = output_dir / "vision-encoder.trt"
        elif module == "text":
            onnx_file = onnx_dir / "text-encoder.onnx"
            engine_file = output_dir / "text-encoder.trt"
        elif module == "decoder":
            onnx_file = onnx_dir / "decoder.onnx"
            engine_file = output_dir / "decoder.trt"

        if not onnx_file.exists():
            print(f"Warning: ONNX file not found: {onnx_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Building {module} encoder...")
        print(f"{'=' * 60}")

        build_engine_from_onnx(
            str(onnx_file),
            str(engine_file),
            opt_batch_size=args.opt_batch_size,
            max_batch_size=args.max_batch_size,
            min_batch_size=args.min_batch_size,
            fp16=args.fp16,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
