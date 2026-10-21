from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from lwcc import LWCC


def _load_csrnet_model():
    """Load the pretrained CSRNet model through LWCC."""

    model = LWCC.load_model(
        model_name="CSRNet",
        model_weights="SHA",
    )

    model.eval()
    model.cpu()

    return model


def analyze_image(image_path):
    """
    Return the estimated crowd count and resized density map.

    Returns:
        count: Rounded estimated crowd count.
        density_map: 2D NumPy array for heatmap visualization.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size

    # FIX: Your LWCC version requires model_name.
    # It returns (image_tensor, additional_value).
    image_tensor, _ = LWCC.load_image(
        str(image_path),
        model_name="CSRNet",
    )

    model = _load_csrnet_model()

    with torch.inference_mode():
        output = model(image_tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim != 4:
        raise RuntimeError(
            f"Unexpected CSRNet output shape: {tuple(output.shape)}"
        )

    density = output[0, 0].detach().cpu().numpy()

    if not np.isfinite(density).all():
        raise RuntimeError("Density map contains invalid values.")

    # Estimate the count from the original model output.
    predicted_count = max(
        0,
        int(round(float(density.sum()))),
    )

    # Resize for visualization only; do not recalculate the count.
    density_tensor = torch.from_numpy(density).float()[None, None]

    display_density = F.interpolate(
        density_tensor,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    return predicted_count, display_density


def count_people(image_path):
    """Backwards-compatible crowd-counting function."""

    count, _ = analyze_image(image_path)
    return count