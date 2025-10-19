import numpy as np
import numpy.typing as npt
from scipy.spatial import distance


def group_distance(group1_coords: npt.ArrayLike, group2_coords: npt.ArrayLike) -> float:
    return np.min(distance.cdist(group1_coords, group2_coords))  # type: ignore
