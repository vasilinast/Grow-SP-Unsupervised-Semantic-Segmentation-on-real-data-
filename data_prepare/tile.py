import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import laspy


def recursive_split(x_min, y_min, x_max, y_max, max_x_size, max_y_size):
    x_size = x_max - x_min
    y_size = y_max - y_min

    if x_size > max_x_size:
        left = recursive_split(
            x_min, y_min, x_min + (x_size // 2), y_max, max_x_size, max_y_size
        )
        right = recursive_split(
            x_min + (x_size // 2), y_min, x_max, y_max, max_x_size, max_y_size
        )
        return left + right
    elif y_size > max_y_size:
        up = recursive_split(
            x_min, y_min, x_max, y_min + (y_size // 2), max_x_size, max_y_size
        )
        down = recursive_split(
            x_min, y_min + (y_size // 2), x_max, y_max, max_x_size, max_y_size
        )
        return up + down
    else:
        return [(x_min, y_min, x_max, y_max)]


def tuple_size(size_string):
    try:
        return tuple(map(float, size_string.split("x")))
    except:
        raise ValueError("Size must be in the form of numberxnumber, e.g., 50.0x65.14")


def tile(
    input_folder: str,
    output_dir: str,
    size: str,
    points_per_iter: int = 10**6,
    crop_min_x: Optional[float] = None,
    crop_min_y: Optional[float] = None,
    crop_max_x: Optional[float] = None,
    crop_max_y: Optional[float] = None,
):
    
    max_size = tuple_size(size)

    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all .laz files in the input folder
    for input_file in input_folder.glob("*.laz"):
        print(f"Processing {input_file.name}...")

        with laspy.open(input_file) as file:

            # Adjust bounds for cropping
            x_min = file.header.x_min + crop_min_x if crop_min_x is not None else file.header.x_min
            y_min = file.header.y_min + crop_min_y if crop_min_y is not None else file.header.y_min
            x_max = file.header.x_max + crop_max_x if crop_max_x is not None else file.header.x_max
            y_max = file.header.y_max + crop_max_y if crop_max_y is not None else file.header.y_max

            sub_bounds = recursive_split(
                x_min,
                y_min,
                x_max,
                y_max,
                max_size[0],
                max_size[1],
            )


            writers: List[Optional[laspy.LasWriter]] = [None] * len(sub_bounds)
            try:
                count = 0
                for points in file.chunk_iterator(points_per_iter):
                    print(f"{count / file.header.point_count * 100:.2f}%")

                    # For performance we need to use copy
                    # so that the underlying arrays are contiguous
                    x, y = points.x.copy(), points.y.copy()

                    point_piped = 0

                    for i, (x_min, y_min, x_max, y_max) in enumerate(sub_bounds):
                        mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

                        if np.any(mask):
                            if writers[i] is None:
                                base_name = input_file.stem
                                output_path = output_dir / f"{base_name}_{i}.laz"
                                writers[i] = laspy.open(
                                    output_path, mode="w", header=file.header
                                )
                            sub_points = points[mask]
                            writers[i].write_points(sub_points)

                        point_piped += np.sum(mask)
                        if point_piped == len(points):
                            break
                    count += len(points)
                print(f"{count / file.header.point_count * 100:.2f}%")
            finally:
                for writer in writers:
                    if writer is not None:
                        writer.close()


if __name__ == "__main__":
    # Example of how to call tile programmatically
    # Replace these values with your inputs
    input_file = 'raw_data_const_site/tile_test'
    output_dir = 'raw_data_const_site/tile_test/output'
    size = "55x58"
    points_per_iter = 10**6

    tile(input_file, output_dir, size, points_per_iter, crop_min_y=18)
