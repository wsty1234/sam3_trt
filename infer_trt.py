"""SAM3 TensorRT Inference Script (v2)

This script provides the same functionality as infer.py but uses TensorRT 10.x
instead of ONNX Runtime for model inference.
"""

import argparse
from pathlib import Path
from typing import Optional
from queue import Queue, Empty
from threading import Thread, Event
import threading

import cv2
import numpy as np
import time
from tokenizers import Tokenizer

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import OrderedDict

# Register built-in TensorRT plugins
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")


class AsyncVisualizer:
    def __init__(self, output_dir: Path):
        self.queue = Queue()
        self.stop_event = Event()
        self.worker = None
        self.output_dir = output_dir

    def _worker(self):
        while True:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break
                self._save_mask_visualization(task)
            except Empty:
                if self.stop_event.is_set():
                    break

    def _save_mask_visualization(self, task):
        img_path, orig_size, result = task
        h, w = orig_size
        black_bg = np.zeros((h, w, 3), dtype=np.uint8)
        color = (30, 144, 255)

        for mask in result["masks"]:
            mask_resized = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR
            )
            mask_bool = mask_resized > 0
            black_bg[mask_bool] = color

            mask_uint8 = mask_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(black_bg, contours, -1, color, 2)

        output_path = self.output_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(output_path), black_bg)
        print(f"  [{img_path.stem}] Saved mask visualization to {output_path}")

    def start(self):
        self.worker = Thread(target=self._worker)
        self.worker.daemon = True
        self.worker.start()

    def submit(self, img_path, orig_size, result):
        self.queue.put((img_path, orig_size, result))

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)
        if self.worker:
            self.worker.join(timeout=5)


class AsyncDataLoader:
    def __init__(self, image_files: list[Path], batch_size: int, queue_size: int):
        self.image_files = image_files
        self.batch_size = batch_size
        self.queue = Queue(maxsize=queue_size)
        self.stop_event = Event()
        self.preprocess_thread = None
        self.total_batches = (len(image_files) + batch_size - 1) // batch_size
        self.processed_batches = 0

    def _worker(self, engine):
        for i in range(0, len(self.image_files), self.batch_size):
            if self.stop_event.is_set():
                break

            batch_files = self.image_files[i : i + self.batch_size]

            images = []
            image_bgr_list = []
            valid_indices = []
            actual_files = []

            for idx, img_path in enumerate(batch_files):
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    continue
                image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                pixel_values, orig_size = engine.preprocess_image(image)
                images.append(pixel_values[0])
                image_bgr_list.append(image_bgr)
                valid_indices.append(idx)
                actual_files.append(img_path)

            if len(images) == 0:
                continue

            pixel_values_batch = np.stack(images, axis=0)
            orig_sizes = []
            for image_bgr in image_bgr_list:
                orig_sizes.append((image_bgr.shape[0], image_bgr.shape[1]))

            batch_data = {
                "pixel_values": pixel_values_batch,
                "image_bgr_list": image_bgr_list,
                "files": actual_files,
                "orig_sizes": orig_sizes,
            }

            self.queue.put(batch_data)
            self.processed_batches += 1

        self.queue.put(None)

    def start(self, engine):
        self.preprocess_thread = Thread(target=self._worker, args=(engine,))
        self.preprocess_thread.daemon = True
        self.preprocess_thread.start()

    def get_batch(self):
        while True:
            try:
                batch_data = self.queue.get(timeout=1)
                if batch_data is None:
                    return None
                return batch_data
            except Empty:
                if self.stop_event.is_set():
                    return None

    def stop(self):
        self.stop_event.set()
        if self.preprocess_thread:
            self.preprocess_thread.join(timeout=5)


class TensorRTInference:
    """Helper class for TensorRT inference"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Create CUDA stream
        self.stream = cuda.Stream()

        # Get IO tensor info for TensorRT 10.x
        self.num_io_tensors = self.engine.num_io_tensors
        self.tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.num_io_tensors)
        ]

        # Determine input/output tensors
        self.input_names = []
        self.output_names = []
        for name in self.tensor_names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        # Store tensor shapes and dtypes
        self.tensor_shapes = {}
        self.tensor_dtypes = {}
        for name in self.tensor_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype

        # Pre-allocate device memory pool with maximum possible size
        self.device_buffers = {}
        self._allocate_memory_pool()

    def _allocate_memory_pool(self):
        """Pre-allocate GPU memory pool with maximum possible size"""
        max_batch_size = 8
        for name in self.tensor_names:
            shape = list(self.tensor_shapes[name])
            dtype = self.tensor_dtypes[name]

            if -1 in shape:
                dim_idx = shape.index(-1)
                if dim_idx == 0 or (dim_idx > 0 and shape[dim_idx - 1] == -1):
                    max_size = [max_batch_size if s == -1 else s for s in shape]
                else:
                    max_size = [s if s != -1 else 1 for s in shape]
            else:
                max_size = shape

            max_size_bytes = int(np.prod(max_size)) * np.dtype(dtype).itemsize
            self.device_buffers[name] = cuda.mem_alloc(max_size_bytes)

    def infer(self, inputs: dict) -> dict:
        """Run inference with given inputs"""
        host_outputs = {}

        # Process inputs first - set shapes
        for name, data in inputs.items():
            if name not in self.input_names:
                continue

            if not data.flags["C_CONTIGUOUS"]:
                data = np.ascontiguousarray(data)

            shape = list(data.shape)
            dtype = self.tensor_dtypes[name]
            size = int(np.prod(shape))

            # Set input shape for dynamic tensors
            self.context.set_input_shape(name, shape)

            # Copy data with pinned memory
            pinned_data = np.ascontiguousarray(data, dtype=dtype)
            cuda.memcpy_htod_async(self.device_buffers[name], pinned_data, self.stream)

            # Set tensor address
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        # Set output shapes and addresses
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = self.tensor_dtypes[name]
            size = int(np.prod(shape))

            # Convert Dims to tuple for numpy/pinned memory
            shape_tuple = tuple(shape)

            # Allocate pinned memory for output
            host_outputs[name] = cuda.pagelocked_empty(shape_tuple, dtype)

            # Set tensor address
            self.context.set_tensor_address(name, int(self.device_buffers[name]))

        # Execute async
        self.context.execute_async_v3(stream_handle=int(self.stream.handle))

        # Copy outputs back to host async
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                host_outputs[name], self.device_buffers[name], self.stream
            )

        # Synchronize for completion
        self.stream.synchronize()

        return host_outputs


def parse_box_prompts(box_str: str) -> tuple[list, list]:
    boxes, labels = [], []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    result = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        result.append([cx, cy, nw, nh])
    return np.array(result, dtype=np.float32)


def xywh_to_xyxy(boxes: list) -> np.ndarray:
    arr = np.array(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


class Sam3TensorRTInferenceV2:
    def __init__(
        self,
        vision_encoder_path: str,
        text_encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        image_height: int = 504,
        image_width: int = 896,
    ):
        self.image_height = image_height
        self.image_width = image_width

        # Load TensorRT engines
        print(f"Loading vision encoder from {vision_encoder_path}")
        self.vision_encoder = TensorRTInference(vision_encoder_path)

        print(f"Loading text encoder from {text_encoder_path}")
        self.text_encoder = TensorRTInference(text_encoder_path)

        print(f"Loading decoder from {decoder_path}")
        self.decoder = TensorRTInference(decoder_path)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)

        # Pre-encode empty text features to avoid repeated inference
        self._empty_text_features, self._empty_text_mask = self._encode_empty_text()

    def _encode_empty_text(self) -> tuple[np.ndarray, np.ndarray]:
        pad_ids = np.full((1, 32), 49407, dtype=np.int64)
        pad_mask = np.zeros((1, 32), dtype=np.int64)
        pad_mask[0, 0] = 1
        inputs = {"input_ids": pad_ids, "attention_mask": pad_mask}
        outputs = self.text_encoder.infer(inputs)
        output_names = list(outputs.keys())
        return outputs[output_names[0]], outputs[output_names[1]]

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        orig_size = (image.shape[0], image.shape[1])
        # Use OpenCV for faster resize
        resized = cv2.resize(
            image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]
        return tensor, orig_size

    def encode_image(self, pixel_values: np.ndarray) -> dict:
        inputs = {"images": pixel_values}
        outputs = self.vision_encoder.infer(inputs)

        output_names = list(outputs.keys())
        return {
            "fpn_feat_0": outputs[output_names[0]],
            "fpn_feat_1": outputs[output_names[1]],
            "fpn_feat_2": outputs[output_names[2]],
            "fpn_pos_2": outputs[output_names[3]],
        }

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.text_encoder.infer(inputs)
        output_names = list(outputs.keys())
        return outputs[output_names[0]], outputs[output_names[1]]

    def decode(
        self,
        vision_features: dict,
        text_features: np.ndarray,
        text_mask: np.ndarray,
        input_boxes: np.ndarray,
        input_boxes_labels: np.ndarray,
    ) -> dict:
        inputs = {
            "fpn_feat_0": vision_features["fpn_feat_0"],
            "fpn_feat_1": vision_features["fpn_feat_1"],
            "fpn_feat_2": vision_features["fpn_feat_2"],
            "fpn_pos_2": vision_features["fpn_pos_2"],
            "text_features": text_features,
            "text_mask": text_mask,
            "input_boxes": input_boxes,
            "input_boxes_labels": input_boxes_labels,
        }

        outputs = self.decoder.infer(inputs)
        output_names = list(outputs.keys())
        return {
            "pred_masks": outputs[output_names[0]],
            "pred_boxes": outputs[output_names[1]],
            "pred_logits": outputs[output_names[2]],
            "presence_logits": outputs[output_names[3]],
        }

    def preprocess_images_batch(
        self, images: list[np.ndarray]
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        all_pixel_values = []
        all_orig_sizes = []

        for image in images:
            pixel_values, orig_size = self.preprocess_image(image)
            all_pixel_values.append(pixel_values[0])
            all_orig_sizes.append(orig_size)

        pixel_values_batch = np.stack(all_pixel_values, axis=0)
        return pixel_values_batch, all_orig_sizes

    def predict(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> dict:
        pixel_values, orig_size = self.preprocess_image(image)

        vision_features = self.encode_image(pixel_values)

        if text:
            text_features, text_mask = self.encode_text(text)
        else:
            text_features, text_mask = self._empty_text_features, self._empty_text_mask

        h, w = orig_size
        if boxes and len(boxes) > 0:
            sx = float(self.image_width) / float(w)
            sy = float(self.image_height) / float(h)
            boxes_resized = [
                [x * sx, y * sy, bw * sx, bh * sy] for x, y, bw, bh in boxes
            ]
            boxes_cxcywh = xywh_to_cxcywh_normalized(
                boxes_resized, self.image_width, self.image_height
            )
            input_boxes = boxes_cxcywh.reshape(1, -1, 4).astype(np.float32)
            if box_labels:
                input_boxes_labels = np.array(box_labels, dtype=np.int64).reshape(1, -1)
            else:
                input_boxes_labels = np.ones((1, input_boxes.shape[1]), dtype=np.int64)
        else:
            input_boxes = np.zeros((1, 1, 4), dtype=np.float32)
            input_boxes_labels = np.full((1, 1), -10, dtype=np.int64)

        outputs = self.decode(
            vision_features,
            text_features=text_features,
            text_mask=text_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )
        return self._postprocess(
            outputs,
            orig_size,
            conf_threshold,
            input_boxes_xywh=boxes,
            input_boxes_labels=box_labels,
        )

    def predict_batch_preprocessed(
        self,
        pixel_values_batch: np.ndarray,
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> list[dict]:
        batch_size = pixel_values_batch.shape[0]

        vision_features = self.encode_image(pixel_values_batch)

        if text:
            text_features, text_mask = self.encode_text(text)
            if text_features.shape[0] == 1 and batch_size > 1:
                text_features = np.repeat(text_features, batch_size, axis=0)
                text_mask = np.repeat(text_mask, batch_size, axis=0)
        else:
            text_features, text_mask = self._empty_text_features, self._empty_text_mask
            if batch_size > 1:
                text_features = np.repeat(text_features, batch_size, axis=0)
                text_mask = np.repeat(text_mask, batch_size, axis=0)

        input_boxes = np.zeros((batch_size, 1, 4), dtype=np.float32)
        input_boxes_labels = np.full((batch_size, 1), -10, dtype=np.int64)

        if boxes and len(boxes) > 0:
            boxes_list = []
            labels_list = []
            for i in range(batch_size):
                sx = float(self.image_width) / float(self.image_width)
                sy = float(self.image_height) / float(self.image_height)
                boxes_resized = [
                    [x * sx, y * sy, bw * sx, bh * sy] for x, y, bw, bh in boxes
                ]
                boxes_cxcywh = xywh_to_cxcywh_normalized(
                    boxes_resized, self.image_width, self.image_height
                )
                boxes_list.append(boxes_cxcywh)
                if box_labels:
                    labels_list.append(np.array(box_labels, dtype=np.int64))
                else:
                    labels_list.append(np.ones(len(boxes), dtype=np.int64))

            input_boxes = np.array(boxes_list, dtype=np.float32)
            input_boxes_labels = np.array(labels_list, dtype=np.int64)

        orig_sizes = [(self.image_height, self.image_width)] * batch_size
        outputs = self.decode(
            vision_features,
            text_features=text_features,
            text_mask=text_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )

        results = []
        for i in range(batch_size):
            batch_output = {
                "pred_masks": outputs["pred_masks"][i : i + 1],
                "pred_boxes": outputs["pred_boxes"][i : i + 1],
                "pred_logits": outputs["pred_logits"][i : i + 1],
                "presence_logits": outputs["presence_logits"][i : i + 1],
            }
            result = self._postprocess(
                batch_output,
                orig_sizes[i],
                conf_threshold,
                input_boxes_xywh=boxes,
                input_boxes_labels=box_labels,
            )
            results.append(result)

        return results

    def predict_batch(
        self,
        images: list[np.ndarray],
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> list[dict]:
        batch_size = len(images)
        pixel_values_batch, orig_sizes = self.preprocess_images_batch(images)

        vision_features = self.encode_image(pixel_values_batch)

        if text:
            text_features, text_mask = self.encode_text(text)
            if text_features.shape[0] == 1 and batch_size > 1:
                text_features = np.repeat(text_features, batch_size, axis=0)
                text_mask = np.repeat(text_mask, batch_size, axis=0)
        else:
            text_features, text_mask = self._empty_text_features, self._empty_text_mask
            if batch_size > 1:
                text_features = np.repeat(text_features, batch_size, axis=0)
                text_mask = np.repeat(text_mask, batch_size, axis=0)

        input_boxes = np.zeros((batch_size, 1, 4), dtype=np.float32)
        input_boxes_labels = np.full((batch_size, 1), -10, dtype=np.int64)

        if boxes and len(boxes) > 0:
            boxes_list = []
            labels_list = []
            for i, (orig_h, orig_w) in enumerate(orig_sizes):
                sx = float(self.image_width) / float(orig_w)
                sy = float(self.image_height) / float(orig_h)
                boxes_resized = [
                    [x * sx, y * sy, bw * sx, bh * sy] for x, y, bw, bh in boxes
                ]
                boxes_cxcywh = xywh_to_cxcywh_normalized(
                    boxes_resized, self.image_width, self.image_height
                )
                boxes_list.append(boxes_cxcywh)
                if box_labels:
                    labels_list.append(np.array(box_labels, dtype=np.int64))
                else:
                    labels_list.append(np.ones(len(boxes), dtype=np.int64))

            input_boxes = np.array(boxes_list, dtype=np.float32)
            input_boxes_labels = np.array(labels_list, dtype=np.int64)

        outputs = self.decode(
            vision_features,
            text_features=text_features,
            text_mask=text_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )

        results = []
        for i in range(batch_size):
            batch_output = {
                "pred_masks": outputs["pred_masks"][i : i + 1],
                "pred_boxes": outputs["pred_boxes"][i : i + 1],
                "pred_logits": outputs["pred_logits"][i : i + 1],
                "presence_logits": outputs["presence_logits"][i : i + 1],
            }
            result = self._postprocess(
                batch_output,
                orig_sizes[i],
                conf_threshold,
                input_boxes_xywh=boxes,
                input_boxes_labels=box_labels,
            )
            results.append(result)

        return results

    def _postprocess(
        self,
        outputs: dict,
        orig_size: tuple[int, int],
        conf_threshold: float,
        input_boxes_xywh: Optional[list] = None,
        input_boxes_labels: Optional[list] = None,
        suppress_neg: bool = False,
        suppress_neg_iou: float = 0.3,
    ) -> dict:
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_logits = outputs["pred_logits"][0]
        presence_logits = outputs["presence_logits"][0, 0]

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > conf_threshold

        h, w = orig_size
        masks = []
        for m in pred_masks[keep]:
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_resized > 0)

        boxes = pred_boxes[keep].copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])
        scores = scores[keep]

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "orig_size": orig_size,
        }


def visualize_worker(args):
    image_bgr, result, output_path = args
    vis = image_bgr.copy()
    color = (30, 144, 255)
    alpha = 0.35

    for mask in result["masks"]:
        mask_bool = mask > 0
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

    cv2.imwrite(output_path, vis)
    return output_path


def visualize_results(
    image: np.ndarray, results: dict, output_path: str, alpha: float = 0.35
):
    vis = image.copy()
    color = (30, 144, 255)

    for mask in results["masks"]:
        mask_bool = mask > 0
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

    cv2.imwrite(output_path, vis)
    print(f"Saved output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 TensorRT Inference (v2)")
    parser.add_argument(
        "--image", type=str, required=True, help="Input image path or directory"
    )
    parser.add_argument("--text", type=str, help="Text prompt")
    parser.add_argument(
        "--boxes", type=str, help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="tensorrt-models-v2",
        help="TensorRT engines directory",
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument(
        "--output", type=str, default="output-trt-v2", help="Output directory"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--image-height", type=int, default=1008)
    parser.add_argument("--image-width", type=int, default=1008)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    args = parser.parse_args()

    if not args.text and not args.boxes:
        parser.error("Please specify --text or --boxes")

    model_dir = Path(args.model_dir)
    engine = Sam3TensorRTInferenceV2(
        vision_encoder_path=str(model_dir / "vision-encoder.trt"),
        text_encoder_path=str(model_dir / "text-encoder.trt"),
        decoder_path=str(model_dir / "decoder.trt"),
        tokenizer_path=args.tokenizer,
        image_height=args.image_height,
        image_width=args.image_width,
    )

    print("Loaded TensorRT models")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    else:
        image_files = [image_path]

    if len(image_files) == 0:
        print("No image files found")
        return

    boxes, box_labels = None, None
    if args.boxes:
        boxes, box_labels = parse_box_prompts(args.boxes)

    if args.batch_size > 1:
        print(f"Running async batch inference with batch size {args.batch_size}")
        queue_size = 4
        async_loader = AsyncDataLoader(image_files, args.batch_size, queue_size)
        async_loader.start(engine)

        async_visualizer = AsyncVisualizer(output_dir)
        async_visualizer.start()

        batch_idx = 0
        total_images = 0
        total_inference_time = 0

        while True:
            batch_data = async_loader.get_batch()
            if batch_data is None:
                break

            actual_batch_size = len(batch_data["image_bgr_list"])
            batch_idx += 1
            total_images += actual_batch_size

            print(
                f"Processing batch {batch_idx}: {actual_batch_size} images (queue size: {async_loader.queue.qsize()}/{queue_size})"
            )

            start = time.time()
            results_list = engine.predict_batch_preprocessed(
                batch_data["pixel_values"],
                text=args.text,
                boxes=boxes,
                box_labels=box_labels,
                conf_threshold=args.conf,
            )
            elapsed = (time.time() - start) * 1000
            total_inference_time += elapsed
            print(
                f"Batch inference time: {elapsed:.2f} ms ({elapsed / actual_batch_size:.2f} ms per image)"
            )

            for j, result in enumerate(results_list):
                img_path = batch_data["files"][j]
                orig_size = batch_data["orig_sizes"][j]
                print(f"  [{img_path.stem}] Found {len(result['masks'])} objects")

                async_visualizer.submit(img_path, orig_size, result)

        async_loader.stop()
        async_visualizer.stop()
        print(
            f"\nTotal inference time: {total_inference_time:.2f} ms ({total_inference_time / total_images:.2f} ms per image)"
        )
    else:
        for img_path in image_files:
            print(f"Processing: {img_path}")
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print(f"Warning: Cannot load image: {img_path}")
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            start = time.time()
            results = engine.predict(
                image,
                text=args.text,
                boxes=boxes,
                box_labels=box_labels,
                conf_threshold=args.conf,
            )
            elapsed = (time.time() - start) * 1000
            print(f"Inference time: {elapsed:.2f} ms")
            print(f"Found {len(results['masks'])} objects")

            output_path = output_dir / f"{img_path.stem}.png"
            visualize_results(image_bgr, results, str(output_path))


if __name__ == "__main__":
    main()
