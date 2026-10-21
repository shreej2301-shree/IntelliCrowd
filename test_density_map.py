from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without requiring a display window

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from lwcc import LWCC


IMAGE_PATH = Path(r"D:\jhu_crowd_v2.0\train\images\0018.jpg")
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "crowd_density_map.png"


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading CSRNet...")
    model = LWCC.load_model(
        model_name="CSRNet",
        model_weights="SHA",
    )
    model.eval()

    print("Loading image...")
    image_tensor, _ = LWCC.load_image(
        str(IMAGE_PATH),
        model_name="CSRNet",
    )

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    print("Generating density map...")
    with torch.no_grad():
        output = model(image_tensor)

    density = output[0, 0].detach().cpu().numpy()
    estimated_count = float(density.sum())

    # Clamp negative values for visualization only.
    density_for_display = np.maximum(density, 0)

    print(f"Estimated count: {estimated_count:.2f}")
    print(f"Density map shape: {density.shape}")

    with Image.open(IMAGE_PATH) as source:
        original = source.convert("RGB")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        heatmap = axes[1].imshow(
            density_for_display,
            cmap="jet",
            interpolation="bilinear",
        )
        axes[1].set_title("CSRNet Crowd Density Map")
        axes[1].axis("off")
        fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(f"IntelliCrowd | Estimated Count: {round(estimated_count)}")
        fig.tight_layout()

        fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not OUTPUT_PATH.is_file():
        raise RuntimeError("Image was not created.")

    print(f"SUCCESS! Density map saved to:\n{OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()