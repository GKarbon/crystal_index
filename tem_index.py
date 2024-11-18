from typing import Literal
from itertools import permutations, product, combinations
from numpy import sqrt, pi, cross, arccos, clip, linalg, dot
# ruff: noqa: E741


class Crystal:
    """
    A class to represent a crystal.

    Attributes
    ----------
    `HKL_LIST` : list of tuple
        The list of (h, k, l) tuples for different faces.
    `crystal_system` : Literal["FCC", "BCC", "SC"]
        The crystal system type.
    `order` : int
        The order of the crystal.
    `hkl_list` : list of tuple
        The list of (h, k, l) tuples representing the faces

    Methods
    -------
    angle_between(v1, v2)
        Returns the angle in radians between vectors `v1` and `v2`.
    get_faces()
        Get the list of faces (hkl values) for the crystal system.
    get_d_spacing(h, k, l)
        Calculate the d-spacing for given Miller indices.
    get_space_group_with_hkl(h, k, l)
        Generate all permutations and sign variations of the given Miller indices.
    get_d_ratio()
        Calculate the ratio of d-spacings for all pairs of faces.
    find_pairs(space_ratio, angle)
        Find pairs of faces that match the given space ratio and angle.
    """

    HKL_LIST = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (2, 1, 0),
        (2, 1, 1),
        (2, 2, 0),
        (3, 0, 0),
        (3, 1, 0),
        (3, 1, 1),
        (2, 2, 2),
        (3, 2, 0),
        (3, 2, 1),
        (4, 0, 0),
        (4, 1, 0),  # 17
        (3, 2, 2),  # 17
        (3, 3, 1),
        (4, 2, 0),
    ]

    def __init__(self, crystal_system: Literal["FCC", "BCC", "SC"], order: int):
        """
        Initialize a Crystal object.

        Parameters
        ----------
        `crystal_system` : Literal["FCC", "BCC", "SC"]
            The crystal system type.
        `order` : int
            The order of the crystal.
        """
        self.crystal_system = crystal_system
        self.order = order
        self.hkl_list = self.get_faces()

    def angle_between(self, v1, v2):
        """
        Returns the angle in radians between vectors `v1` and `v2`.

        Parameters
        ----------
        `v1` : array-like
            The first vector.
        `v2` : array-like
            The second vector.

        Returns
        -------
        float
            The angle in radians.
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

        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return arccos(clip(dot(v1_u, v2_u), -1.0, 1.0))

    def get_faces(self):
        """
        Get the list of faces (hkl values) for the crystal system.

        Returns
        -------
        list of tuple
            The list of (h, k, l) tuples representing the faces.
        """

        def is_even(num: int) -> bool:
            return num % 2 == 0

        hkl_list = []
        match self.crystal_system:
            case "SC":
                for h, k, l in self.HKL_LIST:
                    if len(hkl_list) == self.order:
                        return hkl_list
                    hkl_list.append((h, k, l))
            case "BCC":
                for h, k, l in self.HKL_LIST:
                    if is_even(h + k + l):
                        hkl_list.append((h, k, l))
                    if len(hkl_list) == self.order:
                        return hkl_list
            case "FCC":
                for h, k, l in self.HKL_LIST:
                    if all([is_even(h), is_even(k), is_even(l)]) or all(
                        [not is_even(h), not is_even(k), not is_even(l)]
                    ):
                        hkl_list.append((h, k, l))
                    if len(hkl_list) == self.order:
                        return hkl_list

    def get_d_spacing(self, h: int, k: int, l: int) -> float:
        """
        Calculate the d-spacing for given Miller indices.

        Parameters
        ----------
        `h` : int
            The Miller index h.
        `k` : int
            The Miller index k.
        `l` : int
            The Miller index l.

        Returns
        -------
        float
            The d-spacing value.
        """
        return 1 / sqrt(h**2 + k**2 + l**2)

    def get_space_group_with_hkl(self, h, k, l):
        """
        Generate all permutations and sign variations of the given Miller indices.

        Parameters
        ----------
        `h` : int
            The Miller index h.
        `k` : int
            The Miller index k.
        `l` : int
            The Miller index l.

        Returns
        -------
        list of tuple
            The list of all (h, k, l) permutations and sign variations.
        """

        def generate_hkl_variants(h, k, l):
            variants = []
            signs = list(product([1, -1], repeat=3))
            for sign in signs:
                variants.append((h * sign[0], k * sign[1], l * sign[2]))
            return variants

        result = []
        for hkl in generate_hkl_variants(h, k, l):
            result.extend(permutations(hkl, 3))
        return result

    def get_d_ratio(self):
        """
        Calculate the ratio of d-spacings for all pairs of faces.

        Returns
        -------
        list of tuple
            The list of tuples containing the d-spacing ratio and the corresponding face pairs.
        """
        return [
            (self.get_d_spacing(*pair[0]) / self.get_d_spacing(*pair[1]), pair)
            for pair in combinations(self.hkl_list, 2)
        ]

    def find_pairs(self, space_ratio: float, angle: float):
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
            vector_2_lst = self.get_space_group_with_hkl(*pair[1])
            for vector_2 in vector_2_lst:
                if self.angle_between(vector_1, vector_2) * 180 / pi < angle:
                    print(f"Vector 1: {vector_1}, Vector 2: {vector_2}")
                    print(f"Zone axis: {cross(vector_1, vector_2)}")
                    print("*" * 30)
                    break


def main():
    """
    Main function to create a Crystal object and find pairs of faces.
    """
    crystal = Crystal("FCC", 8)
    crystal.find_pairs(28.93 / 11.11, 66)


if __name__ == "__main__":
    main()