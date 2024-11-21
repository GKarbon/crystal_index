from enum import Enum
from typing import Literal, Union
from itertools import combinations, combinations_with_replacement, permutations, product
from numpy import arccos, array, clip, cross, dot, linalg, newaxis, rad2deg, square
# ruff: noqa: E741


def get_possible_index_combinations(
    largest_miller_index: int,
) -> list[tuple[int, int, int]]:
    """
    Generate all possible index combinations for a given largest_miller_index.

    Parameters
    ----------
    `largest_miller_index` : int
        The largest_miller_index of the combinations.

    Returns
    -------
    list of tuple
        The list of all possible (h, k, l) index combinations.
    """

    def sort_by_square_sum(t) -> int:
        return sum(square(t))

    return sorted(
        combinations_with_replacement(range(largest_miller_index + 1), 3),
        key=sort_by_square_sum,
    )[1:]


class CrystalSystem(Enum):
    """
    Enum to represent different crystal systems.

    Attributes
    ----------
    `FCC` : str
        Face-Centered Cubic crystal system.
    `BCC` : str
        Body-Centered Cubic crystal system.
    `SC` : str
        Simple Cubic crystal system.
    """

    FCC = "FCC"
    BCC = "BCC"
    SC = "SC"


class CrystalPlane:
    """
    A class to represent a crystal plane.

    Attributes
    ----------
    `h` : int
        The Miller index h.
    `k` : int
        The Miller index k.
    `l` : int
        The Miller index l.
    `planenormal` : numpy.ndarray
        The normal vector of the plane.

    Methods
    -------
    angle_between(p1, p2)
        Returns the angle in radians between plane `p1` and `p2`.
    get_zone_axis(p1, p2)
        Returns the zone axis for the given planes.
    get_d_spacing()
        Calculate the d-spacing for given Miller indices.
    get_equivalent_planes()
        Generate all permutations and sign variations of the given Miller indices.
    """

    def __init__(self, h, k, l):
        self.h = h
        self.k = k
        self.l = l
        self.planenormal = array([h, k, l])

    def __str__(self) -> str:
        return f"({self.h}, {self.k}, {self.l})"

    @classmethod
    def angle_between(cls, p1: "CrystalPlane", p2: "CrystalPlane"):
        """
        Returns the angle in radians between plane `p1` and `p2`.

        Parameters
        ----------
        `p1` : CrystalPlane
            The first plane.
        `p2` : CrystalPlane
            The second plane.

        Returns
        -------
        float
            The angle in angles.
        """

        def unit_vector(vector):
            """
            Returns the unit vector of the vector.

            Parameters
            ----------
            `vector` : array-like
                The input vector.

            Returns
            -------
            array-like
                The unit vector.
            """
            return vector / linalg.norm(vector)

        p1_u = unit_vector(p1.planenormal)
        p2_u = unit_vector(p2.planenormal)
        return rad2deg(arccos(clip(dot(p1_u, p2_u), -1.0, 1.0)))

    @classmethod
    def get_zone_axis(cls, p1: "CrystalPlane", p2: "CrystalPlane"):
        """
        Returns the zone axis for the given planes.

        Parameters
        ----------
        `p1` : CrystalPlane
            The first plane.
        `p2` : CrystalPlane
            The second plane.

        Returns
        -------
        numpy.ndarray
            The zone axis vector.
        """
        return cross(p1.planenormal, p2.planenormal)

    def get_d_spacing(self) -> float:
        """
        Calculate the d-spacing for given Miller indices.

        Returns
        -------
        float
            The d-spacing value.
        """
        return 1 / linalg.norm(self.planenormal)

    def get_equivalent_planes(self) -> list["CrystalPlane"]:
        """
        Generate all permutations and sign variations of the given Miller indices.

        Returns
        -------
        list of tuple
            The list of all (h, k, l) permutations and sign variations.
        """

        possible_miller_indices = array(list(permutations([self.h, self.k, self.l])))
        sign_variants = array(list(product([1, -1], repeat=3)))
        possible_planes = (
            possible_miller_indices[:, newaxis, :] * sign_variants
        ).reshape(-1, 3)
        return [CrystalPlane(*plane) for plane in possible_planes]


class Crystal:
    """
    A class to represent a crystal.

    Attributes
    ----------
    `HKL_LIST` : list of tuple
        The list of (h, k, l) tuples for different faces.
    `crystal_system` : Union[CrystalSystem, Literal["FCC", "BCC", "SC"]]
        The crystal system type.
    `order` : int
        The order of the crystal.
    `planes` : list of CrystalPlane
        The list of CrystalPlane objects representing the faces.

    Methods
    -------
    get_planes()
        Get the list of faces (hkl values) for the crystal system.
    get_d_ratio()
        Calculate the ratio of d-spacings for all pairs of faces.
    find_pairs(space_ratio, angle)
        Find pairs of faces that match the given space ratio and angle.
    """

    HKL_LIST = get_possible_index_combinations(5)

    def __init__(
        self,
        crystal_system: Union[CrystalSystem, Literal["FCC", "BCC", "SC"]],
        order: int,
    ) -> None:
        """
        Initialize a Crystal object.

        Parameters
        ----------
        `crystal_system` : Union[CrystalSystem, Literal["FCC", "BCC", "SC"]]
            The crystal system type.
        `order` : int
            The order of the crystal.
        """
        self.crystal_system = (
            crystal_system
            if isinstance(crystal_system, CrystalSystem)
            else CrystalSystem(crystal_system)
        )
        self.order = order
        self.planes = self.get_planes()

    def get_planes(self) -> list[CrystalPlane]:
        """
        Get the list of faces (hkl values) for the crystal system.

        Returns
        -------
        list of CrystalPlane
            The list of CrystalPlane objects representing the faces.
        """

        def is_even(num: int) -> bool:
            return num % 2 == 0

        hkl_list = []
        match self.crystal_system:
            case CrystalSystem.SC:
                for h, k, l in self.HKL_LIST:
                    if len(hkl_list) == self.order:
                        return hkl_list
                    hkl_list.append(CrystalPlane(h, k, l))
            case CrystalSystem.BCC:
                for h, k, l in self.HKL_LIST:
                    if is_even(h + k + l):
                        hkl_list.append(CrystalPlane(h, k, l))
                    if len(hkl_list) == self.order:
                        return hkl_list
            case CrystalSystem.FCC:
                for h, k, l in self.HKL_LIST:
                    if all([is_even(h), is_even(k), is_even(l)]) or all(
                        [not is_even(h), not is_even(k), not is_even(l)]
                    ):
                        hkl_list.append(CrystalPlane(h, k, l))
                    if len(hkl_list) == self.order:
                        return hkl_list

    def get_d_ratio(self) -> list[tuple[float, tuple[CrystalPlane, CrystalPlane]]]:
        """
        Calculate the ratio of d-spacings for all pairs of faces.

        Returns
        -------
        list of tuple
            The list of tuples containing the d-spacing ratio and the corresponding face pairs.
        """
        return [
            (pair[0].get_d_spacing() / pair[1].get_d_spacing(), pair)
            for pair in combinations(self.planes, 2)
        ]

    def find_pairs(self, space_ratio: float, angle: float) -> None:
        """
        Find pairs of faces that match the given space ratio and angle.

        Parameters
        ----------
        `space_ratio` : float
            The target space ratio.
        `angle` : float
            The target angle in degrees.
        """
        lower_limit = space_ratio * 0.8
        higher_limit = space_ratio * 1.2
        possible_pairs = [
            i[1] for i in self.get_d_ratio() if lower_limit < i[0] < higher_limit
        ]
        for pair in possible_pairs:
            vector_1 = pair[0]
            vector_2_lst = pair[1].get_equivalent_planes()
            for vector_2 in vector_2_lst:
                if CrystalPlane.angle_between(vector_1, vector_2) < angle:
                    print(f"Vector 1: {vector_1}, Vector 2: {vector_2}")
                    print(
                        f"Zone axis: {CrystalPlane.get_zone_axis(vector_1, vector_2)}"
                    )
                    print("*" * 30)
                    break


def main() -> None:
    """
    Main function to create a Crystal object and find pairs of faces.
    """
    crystal = Crystal("FCC", 8)  # Accepts string input
    crystal.find_pairs(
        28.93 / 11.11, 66
    )  # TODO: Input two lengths(float), sort in the function to calculate


if __name__ == "__main__":
    main()
