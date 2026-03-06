import numpy as np


def place_proteins_in_grid(
    residue_positions: list[tuple[float, float, float]],
    n_proteins_x: int,
    n_proteins_y: int,
    n_proteins_z: int,
    grid_buffer: float,
    n_proteins_max: int | None = None,
) -> list[np.ndarray]:
    positions_single_proteins = np.array(residue_positions)

    if n_proteins_max is not None:
        assert n_proteins_max <= n_proteins_x * n_proteins_y * n_proteins_z

    x_coords = [r[0] for r in positions_single_proteins]
    y_coords = [r[1] for r in positions_single_proteins]
    z_coords = [r[2] for r in positions_single_proteins]

    span_x = np.max(x_coords) - np.min(x_coords) + grid_buffer
    span_y = np.max(y_coords) - np.min(y_coords) + grid_buffer
    span_z = np.max(z_coords) - np.min(z_coords) + grid_buffer

    n_proteins_added = 0
    total_positions = []
    for ix in range(n_proteins_x):
        for iy in range(n_proteins_y):
            for iz in range(n_proteins_z):
                offset = np.array([ix * span_x, iy * span_y, iz * span_z])
                total_positions.append(positions_single_proteins + offset)
                n_proteins_added += 1

                if n_proteins_max is not None and n_proteins_added == n_proteins_max:
                    return total_positions

    return total_positions
