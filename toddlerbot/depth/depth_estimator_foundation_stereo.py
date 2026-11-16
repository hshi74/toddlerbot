"""TensorRT-accelerated stereo depth estimation using Foundation Stereo model.

This module provides stereo depth estimation capabilities using a TensorRT engine
for high-performance inference with camera calibration and rectification support.
"""

import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch

# Import the required modules (assuming they are available)
try:
    from onnx_tensorrt import tensorrt_engine
except ModuleNotFoundError as e:  # only .engine and .plan files need onnx-tensorrt
    if e.name != "onnx_tensorrt":
        raise
    tensorrt_engine = None
import onnxruntime as ort

from toddlerbot.depth.depth_utils import (
    depth_to_xyzmap,
    get_rectification_maps,
    to_open3d_Cloud,
)
from toddlerbot.utils.misc_utils import log, profile

# Initialize TensorRT logger (like in the reference)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # Use WARNING level like reference
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


def preprocess_image(
    img: np.ndarray, target_height: int, target_width: int
) -> torch.Tensor:
    """Preprocess image like in run_demo_tensorrt.py"""
    # Resize if needed
    if img.shape[0] != target_height or img.shape[1] != target_width:
        img = cv2.resize(img, (target_width, target_height))

    # Convert to torch tensor like in reference: torch.as_tensor(input_image.copy()).float()[None].permute(0,3,1,2).contiguous()
    resized_image = (
        torch.as_tensor(img.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    )
    return resized_image


def get_onnx_model(model_path):
    """Load ONNX model like in run_demo_tensorrt.py"""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(
        model_path, sess_options=session_options, providers=["CUDAExecutionProvider"]
    )
    return model


def get_model_input_shape(model_path):
    """Extract input shape (height, width) from model file"""
    if model_path.endswith(".onnx"):
        # Create a temporary session just to get input shape
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[
            0
        ].shape  # Assuming first input is left image
        # Shape is typically [batch, channels, height, width]
        if len(input_shape) >= 4:
            height, width = input_shape[2], input_shape[3]
            return height, width
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")

    elif model_path.endswith(".engine") or model_path.endswith(".plan"):
        # Extract shape from TensorRT engine without full initialization
        with open(model_path, "rb") as f:
            engine_data = f.read()

        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)

        # Get input tensor shape
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = engine.get_tensor_shape(name)
                # Convert negative dimensions to positive (dynamic shapes)
                shape = [d if d > 0 else 1 for d in shape]
                if len(shape) >= 4:
                    height, width = shape[2], shape[3]
                    return height, width
                break

        raise ValueError("Could not extract input shape from TensorRT engine")

    elif model_path.endswith(".pte"):
        from executorch.runtime import Runtime

        program = Runtime.get().load_program(model_path)
        shape = program.metadata("forward").input_tensor_meta(0).sizes()
        return shape[2], shape[3]
    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def get_engine_model(model_path):
    """Load TensorRT engine model like in run_demo_tensorrt.py"""
    if tensorrt_engine is None:
        raise ImportError("Loading .engine or .plan files needs onnx-tensorrt")
    with open(model_path, "rb") as file:
        engine_data = file.read()
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)
    engine = tensorrt_engine.Engine(engine)
    return engine


def get_pte_model(model_path):
    """Load the forward method of an ExecuTorch program (.pte)."""
    import torch_tensorrt_executorch_runtime  # noqa: F401  (registers the TensorRT backend)
    from executorch.runtime import Runtime

    return Runtime.get().load_program(model_path).load_method("forward")


@dataclass(frozen=True, slots=True)
class DepthResult:
    """Container for all artifacts produced by depth estimation."""

    depth: np.ndarray
    disparity: Optional[np.ndarray] = None
    rectified_left: Optional[np.ndarray] = None
    rectified_right: Optional[np.ndarray] = None
    original_left: Optional[np.ndarray] = None
    original_right: Optional[np.ndarray] = None


class DepthEstimatorFoundationStereo:
    """Depth estimation using Foundation Stereo model."""

    def __init__(
        self,
        calib_params_path,
        rec_params_path,
        engine_path,  # Can be .onnx, .engine, .plan, or .pte file
        calib_width,
        calib_height,
    ):
        # initialize tensorrt engine
        self._init_engine(engine_path)

        # initialize calibration and rectification maps
        self._init_calibration(
            calib_params_path, rec_params_path, calib_width, calib_height
        )

    def _init_engine(self, model_path) -> None:
        """
        Initialize model like in run_demo_tensorrt.py - auto-detect format.
        Args:
            model_path: Path to the model file (.onnx, .engine, .plan, or .pte).
        Returns:
            None
        """
        # Extract input shape from the model file
        self.height, self.width = get_model_input_shape(model_path)
        log(
            f"Extracted model input shape: {self.height}x{self.width}",
            header="DepthEstimator",
        )

        # Auto-detect model format and load like in run_demo_tensorrt.py
        if model_path.endswith(".onnx"):
            self.model = get_onnx_model(model_path)
            self.model_type = "onnx"
            log("ONNX model loaded", header="DepthEstimator")
        elif model_path.endswith(".engine") or model_path.endswith(".plan"):
            self.model = get_engine_model(model_path)
            self.model_type = "engine"
            log("TensorRT engine model loaded", header="DepthEstimator")
        elif model_path.endswith(".pte"):
            self.model = get_pte_model(model_path)
            self.model_type = "pte"
            log("ExecuTorch program loaded", header="DepthEstimator")
        else:
            raise ValueError(
                f"Unknown model format {model_path}. Supported formats: .onnx, .engine, .plan, .pte"
            )

    def _init_calibration(
        self,
        calib_params_path,
        rec_params_path,
        calib_width,
        calib_height,
    ) -> None:
        """
        Initialize calibration and rectification maps.
        Args:
            calib_params_path: Path to the calibration parameters file.
            rec_params_path: Path to the rectification parameters file.
            calib_width: Width used for calibration.
            calib_height: Height used for calibration.
        Returns:
            None
        """
        with open(calib_params_path, "rb") as f:
            calib_params = pickle.load(f)
            self.K1 = calib_params["K1"]
            self.K2 = calib_params["K2"]
            scale_factor = self.width / calib_width
            assert np.isclose(scale_factor, self.height / calib_height), (
                f"Scale factor must be the same for both width and height: {scale_factor} != {self.height / calib_height}"
            )
            # self.scaled_K = self.K1.copy()
            # self.scaled_K[:2] *= scale_factor
            self.T = (calib_params["T"],)
            self.baseline = np.linalg.norm(self.T)
            if self.T[0][0] < 0:
                self.scaled_K = self.K1.copy()
                self.scaled_K[:2] *= scale_factor
            elif self.T[0][0] > 0:
                self.scaled_K = self.K2.copy()
                self.scaled_K[:2] *= scale_factor
            else:
                raise ValueError("Baseline is zero")

        with open(rec_params_path, "rb") as f:
            rec_params = pickle.load(f)

            # P1 and P2 are the same except P2[0,3]
            self.P1 = rec_params["P1"]
            scale_factor = self.width / calib_width
            scaled_rectified_fx = (
                self.P1[0, 0] * scale_factor
            )  # Focal length from P1 matrix
            scaled_rectified_fy = self.P1[1, 1] * scale_factor
            scaled_rectified_cx = self.P1[0, 2] * scale_factor
            scaled_rectified_cy = self.P1[1, 2] * scale_factor
            self.scaled_rectified_K = np.array(
                [
                    [scaled_rectified_fx, 0, scaled_rectified_cx],
                    [0, scaled_rectified_fy, scaled_rectified_cy],
                    [0, 0, 1],
                ]
            )

            # Pre-compute the full numerator for maximum speed in the inference loop
            self.fx_times_baseline = scaled_rectified_fx * self.baseline

        self.map1_left, self.map2_left, self.map1_right, self.map2_right = (
            get_rectification_maps(
                calib_params, rec_params, (calib_width, calib_height)
            )
        )

        # Pre-compute combined rectification and resize maps
        self.combined_map1_left = cv2.resize(
            self.map1_left,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        self.combined_map2_left = cv2.resize(
            self.map2_left,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        self.combined_map1_right = cv2.resize(
            self.map1_right,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        self.combined_map2_right = cv2.resize(
            self.map2_right,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _infer(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """
        Run inference like in run_demo_tensorrt.py.

        Args:
            left_img: Left image as numpy array.
            right_img: Right image as numpy array.
        Returns:
            Disparity map as numpy array.
        """
        # Preprocess images
        left_tensor = preprocess_image(left_img, self.height, self.width)
        right_tensor = preprocess_image(right_img, self.height, self.width)

        # Run inference based on model type, exactly like in run_demo_tensorrt.py
        torch.cuda.synchronize()
        if self.model_type == "onnx":
            left_disp = self.model.run(
                None, {"left": left_tensor.numpy(), "right": right_tensor.numpy()}
            )[0]
        elif self.model_type == "pte":
            left_disp = self.model.execute((left_tensor, right_tensor))[0].numpy()
        else:  # engine or plan
            left_disp = self.model.run([left_tensor.numpy(), right_tensor.numpy()])[0]
        torch.cuda.synchronize()

        return left_disp.squeeze()  # HxW

    def _process_images(self, img0, img1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process images with combined rectification and resize.
        Args:
            img0: Left image.
            img1: Right image.
        Returns:
            Tuple of processed images.
        """
        img0 = cv2.remap(
            img0,
            self.combined_map1_left,
            self.combined_map2_left,
            interpolation=cv2.INTER_LINEAR,
        )
        img1 = cv2.remap(
            img1,
            self.combined_map1_right,
            self.combined_map2_right,
            interpolation=cv2.INTER_LINEAR,
        )

        return img0, img1

    @profile()
    def get_depth(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        *,
        remove_invisible: bool = False,
        return_all: bool = False,
    ) -> DepthResult:
        """Estimate per-pixel depth from the stereo pair.

        Args:
            img_left: Left camera image.
            img_right: Right camera image.
            remove_invisible: If ``True``, mask points that project outside the
                right-camera image.
            return_all: If ``True``, populate every field in :class:`DepthResult`;
                otherwise only ``depth`` is guaranteed to be non-``None``.

        Returns:
            An immutable :class:`DepthResult` whose fields are either populated or
            ``None`` depending on ``return_all``.
        """

        original_left = img_left.copy() if return_all else None
        original_right = img_right.copy() if return_all else None

        # Pre-process
        img_left, img_right = self._process_images(img_left, img_right)

        rectified_left = img_left.copy() if return_all else None
        rectified_right = img_right.copy() if return_all else None

        # Depth inference
        # Switch the order when the sign of the baseline is positive which means right images are rectified to the left side on the baseline
        if self.T[0][0] < 0:
            disparity = self._infer(img_left, img_right)
        elif self.T[0][0] > 0:
            disparity = self._infer(img_right, img_left)
        else:
            raise ValueError("Baseline is zero")

        if remove_invisible:
            xx = np.arange(disparity.shape[1])[None, :]
            invalid = (xx - disparity) < 0
            disparity_masked = disparity.copy()
            disparity_masked[invalid] = np.inf
            disparity = disparity_masked

        # Use the same formula as run_demo_tensorrt.py: depth = K[0,0]*baseline/(left_disp + doffs)
        doffs = 0
        depth: np.ndarray = self.fx_times_baseline / (disparity + doffs)

        return DepthResult(
            depth=depth,
            disparity=disparity if return_all else None,
            rectified_left=rectified_left,
            rectified_right=rectified_right,
            original_left=original_left,
            original_right=original_right,
        )

    def get_pcl(
        self,
        depth,
        resized_image,
        is_BGR=True,
        zmim=0.0,
        zmax=None,
        denoise_cloud=False,
        denoise_nb_points=30,
        denoise_radius=0.03,
    ):
        """
        Convert depth map to point cloud.
        Args:
            depth: Depth map.
            resized_image: Resized image.
            is_BGR: Whether the image is in BGR format.
            zmin: Minimum depth (meters).
            zmax: Maximum depth (meters).
            denoise_cloud: Whether to denoise the point cloud.
            denoise_nb_points: Number of points to use for radius outlier removal.
            denoise_radius: Radius for radius outlier removal.
        Returns:
            Point cloud (open3d.geometry.PointCloud)
        """
        zmax = np.inf if zmax is None else zmax
        # Use the same approach as run_demo_tensorrt.py
        xyz_map = depth_to_xyzmap(depth, self.scaled_rectified_K, zmin=zmim)

        if is_BGR:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        pcd = to_open3d_Cloud(xyz_map.reshape(-1, 3), resized_image.reshape(-1, 3))
        points = np.asarray(pcd.points)
        keep_ids = np.where((points[:, 2] > 0) & (points[:, 2] <= zmax))[0]
        pcd = pcd.select_by_index(keep_ids)

        if denoise_cloud:
            _, ind = pcd.remove_radius_outlier(
                nb_points=denoise_nb_points, radius=denoise_radius
            )
            pcd = pcd.select_by_index(ind)

        return pcd
