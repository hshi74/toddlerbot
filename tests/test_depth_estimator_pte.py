# test_depth_estimator_pte.py

import importlib.util
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch_tensorrt = pytest.importorskip("torch_tensorrt")
# Check without importing: the estimator must register the TensorRT backend itself.
if importlib.util.find_spec("torch_tensorrt_executorch_runtime") is None:
    pytest.skip("needs torch-tensorrt-executorch-runtime", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("the depth estimator needs CUDA", allow_module_level=True)

from toddlerbot.depth.depth_estimator_foundation_stereo import (  # noqa: E402
    DepthEstimatorFoundationStereo,
    preprocess_image,
)

HEIGHT, WIDTH = 96, 128


class FakeStereo(torch.nn.Module):
    """Stands in for FoundationStereo: same inputs and output shape."""

    def forward(self, left, right):
        return (left - 0.5 * right).mean(dim=1, keepdim=True) + 128


def write_calibration(tmp_path):
    """Write pinhole calibration files for an already rectified camera pair."""
    K = np.array([[100.0, 0, WIDTH / 2], [0, 100.0, HEIGHT / 2], [0, 0, 1]])
    baseline = 0.05
    calib = {
        "K1": K,
        "K2": K,
        "D1": np.zeros((4, 1)),
        "D2": np.zeros((4, 1)),
        "R": np.eye(3),
        "T": np.array([[-baseline], [0.0], [0.0]]),
    }
    rec = {
        "R1": np.eye(3),
        "R2": np.eye(3),
        "P1": np.hstack([K, np.zeros((3, 1))]),
        "P2": np.hstack([K, np.array([[-K[0, 0] * baseline], [0.0], [0.0]])]),
        "Q": np.eye(4),
    }
    calib_path, rec_path = tmp_path / "calibration.pkl", tmp_path / "rectification.pkl"
    calib_path.write_bytes(pickle.dumps(calib))
    rec_path.write_bytes(pickle.dumps(rec))
    return str(calib_path), str(rec_path)


def test_pte_matches_pytorch(tmp_path):
    """
    Test that a TensorRT .pte gives the same disparity as the PyTorch model.
    """
    model = FakeStereo().eval()
    example = (
        torch.rand(1, 3, HEIGHT, WIDTH, device="cuda") * 255,
        torch.rand(1, 3, HEIGHT, WIDTH, device="cuda") * 255,
    )
    trt_module = torch_tensorrt.dynamo.compile(
        torch.export.export(model, example), inputs=list(example), min_block_size=1
    )
    pte_path = tmp_path / "fake_stereo.pte"
    torch_tensorrt.save(trt_module, str(pte_path), output_format="executorch")

    calib_path, rec_path = write_calibration(tmp_path)
    estimator = DepthEstimatorFoundationStereo(
        calib_path, rec_path, str(pte_path), calib_width=WIDTH, calib_height=HEIGHT
    )
    assert (estimator.height, estimator.width) == (HEIGHT, WIDTH)

    rng = np.random.default_rng(0)
    left = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    result = estimator.get_depth(left, right, return_all=True)

    expected = model(
        preprocess_image(result.rectified_left, HEIGHT, WIDTH),
        preprocess_image(result.rectified_right, HEIGHT, WIDTH),
    ).squeeze()
    assert isinstance(result.disparity, np.ndarray)
    np.testing.assert_allclose(result.disparity, expected.numpy(), rtol=1e-5, atol=1e-4)
