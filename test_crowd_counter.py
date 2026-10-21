from pathlib import Path

from services.crowd_counter import count_people


def main():
    image_path = Path(
        r"D:\jhu_crowd_v2.0\train\images\0018.jpg"
    )

    try:
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Test image not found: {image_path}"
            )

        print("Loading crowd-counting model...")
        print(f"Test image: {image_path}")

        count = count_people(image_path)

        print("-" * 40)
        print(f"Estimated people count: {count}")
        print("-" * 40)

    except Exception as error:
        print(f"Test failed: {error}")
        raise


if __name__ == "__main__":
    main()