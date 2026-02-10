from pathlib import Path
import shutil

def collect_pngs(
    input_root: str | Path,
    output_dir: str | Path,
    prefix_with_parent: bool = True
):
    """
    Collect all .png files from nested directories into a single directory.

    Args:
        input_root: Root directory containing subdirectories with PNGs.
        output_dir: Directory where all PNGs will be copied.
        prefix_with_parent: If True, prefix filenames with parent folder
                            to avoid name collisions.
    """
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for png_path in input_root.rglob("*.png"):
        if not png_path.is_file():
            continue

        if prefix_with_parent:
            new_name = f"{png_path.parent.name}_{png_path.name}"
        else:
            new_name = png_path.name

        dst = output_dir / new_name

        # Handle accidental name collisions
        counter = 1
        while dst.exists():
            dst = output_dir / f"{dst.stem}_{counter}{dst.suffix}"
            counter += 1

        shutil.copy2(png_path, dst)


collect_pngs(
    input_root="/home/anas/datasets/ssl4eo-small/rgb",
    output_dir="/home/anas/datasets/ssl42eo-small-torun"
)

