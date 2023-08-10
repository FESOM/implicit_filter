import unittest
from implicit_filter._auxiliary import make_tri, neighboring_triangles, neighbouring_nodes
import numpy as np
from numpy.testing import assert_array_equal


class TestMakeTri(unittest.TestCase):
    @staticmethod
    def test_small_case():
        nx = 3
        ny = 3
        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        out = np.array([[0, 1, 3],
                        [1, 4, 3],
                        [1, 2, 4],
                        [2, 5, 4],
                        [3, 4, 6],
                        [4, 7, 6],
                        [4, 5, 7],
                        [5, 8, 7]])

        assert_array_equal(make_tri(nodnum, nx, ny), out)


class TestNeighbouringTriangles(unittest.TestCase):

    def setUp(self) -> None:
        Lx = 5
        Ly = Lx
        dxm = 1
        dym = dxm

        xx = np.arange(0, Lx + 1, dxm)
        yy = np.arange(0, Ly + 1, dym)
        nx = len(xx)
        ny = len(yy)

        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        self.tri = make_tri(nodnum, nx, ny)
        self.n2d = nx * ny
        self.e2d = len(self.tri[:, 1])

    def test_neighboring_triangles(self):
        out_num = np.array([1, 3, 3, 3, 3, 2, 3, 6, 6, 6, 6, 3, 3, 6, 6, 6, 6, 3, 3, 6, 6, 6,
                            6, 3, 3, 6, 6, 6, 6, 3, 2, 3, 3, 3, 3, 1])
        out_pos = np.array([[0, 0, 2, 4, 6, 8, 0, 1, 3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 20,
                             21, 23, 25, 27, 29, 30, 31, 33, 35, 37, 39, 40, 41, 43, 45, 47, 49],
                            [0, 1, 3, 5, 7, 9, 1, 2, 4, 6, 8, 18, 11, 12, 14, 16, 18, 28, 21,
                             22, 24, 26, 28, 38, 31, 32, 34, 36, 38, 48, 41, 42, 44, 46, 48, 0],
                            [0, 2, 4, 6, 8, 0, 10, 3, 5, 7, 9, 19, 20, 13, 15, 17, 19, 29, 30,
                             23, 25, 27, 29, 39, 40, 33, 35, 37, 39, 49, 0, 43, 45, 47, 49, 0],
                            [0, 0, 0, 0, 0, 0, 0, 10, 12, 14, 16, 0, 0, 20, 22, 24, 26, 0, 0,
                             30, 32, 34, 36, 0, 0, 40, 42, 44, 46, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 11, 13, 15, 17, 0, 0, 21, 23, 25, 27, 0, 0,
                             31, 33, 35, 37, 0, 0, 41, 43, 45, 47, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 12, 14, 16, 18, 0, 0, 22, 24, 26, 28, 0, 0,
                             32, 34, 36, 38, 0, 0, 42, 44, 46, 48, 0, 0, 0, 0, 0, 0, 0]])

        num, pos = neighboring_triangles(self.n2d, self.e2d, self.tri)
        assert_array_equal(num, out_num)
        assert_array_equal(pos, out_pos)


class TestNeighbouringElements(unittest.TestCase):

    def setUp(self) -> None:
        Lx = 5
        Ly = Lx
        dxm = 1
        dym = dxm

        xx = np.arange(0, Lx + 1, dxm)
        yy = np.arange(0, Ly + 1, dym)
        nx = len(xx)
        ny = len(yy)

        nodnum = np.arange(0, nx * ny)
        nodnum = np.reshape(nodnum, [ny, nx]).T

        self.tri = make_tri(nodnum, nx, ny)
        self.n2d = nx * ny
        self.e2d = len(self.tri[:, 1])
        self.ne_num, self.ne_pos = neighboring_triangles(self.n2d, self.e2d, self.tri)

    def test_neighboring_elements(self):
        out_num = np.array([3, 5, 5, 5, 5, 4, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 5, 5, 7, 7, 7,
                            7, 5, 4, 5, 5, 5, 5, 3])
        out_pos = np.array([[0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                            [1, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 11, 7, 13, 14, 15,
                             16, 17, 13, 19, 20, 21, 22, 23, 19, 25, 26, 27, 28, 29, 25, 31, 32, 33, 34, 35],
                            [6, 6, 7, 8, 9, 10, 6, 6, 7, 8, 9, 10, 12, 12, 13, 14,
                             15, 16, 18, 18, 19, 20, 21, 22, 24, 24, 25, 26, 27, 28, 30, 30, 31, 32, 33, 34],
                            [0, 7, 8, 9, 10, 11, 7, 2, 3, 4, 5, 16, 13, 8, 9, 10,
                             11, 22, 19, 14, 15, 16, 17, 28, 25, 20, 21, 22, 23, 34, 31, 26, 27, 28, 29, 0],
                            [0, 2, 3, 4, 5, 0, 12, 8, 9, 10, 11, 17, 18, 14, 15, 16,
                             17, 23, 24, 20, 21, 22, 23, 29, 30, 26, 27, 28, 29, 35, 0, 32, 33, 34, 35, 0],
                            [0, 0, 0, 0, 0, 0, 0, 12, 13, 14, 15, 0, 0, 18, 19, 20,
                             21, 0, 0, 24, 25, 26, 27, 0, 0, 30, 31, 32, 33, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 13, 14, 15, 16, 0, 0, 19, 20, 21,
                             22, 0, 0, 25, 26, 27, 28, 0, 0, 31, 32, 33, 34, 0, 0, 0, 0, 0, 0, 0]])

        num, pos = neighbouring_nodes(self.n2d, self.tri, self.ne_num, self.ne_pos)

        assert_array_equal(num, out_num)
        assert_array_equal(pos, out_pos)


if __name__ == '__main__':
    unittest.main()
