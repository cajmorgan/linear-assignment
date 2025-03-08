

from typing import List
from linear_assignment import LinearAssignmentMunkres
from scipy.optimize import linear_sum_assignment
import torch


def test_munkres_min_4x4():
    matrices = []
    for i in range(0, 100):
        matrices.append(torch.randint(0, 100, (4, 4)))

    check_cost_matrices_against_scipy(matrices)


def test_munkres_min_8x8():
    matrices = []
    for i in range(0, 100):
        matrices.append(torch.randint(0, 100, (8, 8)))

    check_cost_matrices_against_scipy(matrices)


def test_munkres_min_16x16():
    matrices = []
    for i in range(0, 100):
        matrices.append(torch.randint(0, 100, (16, 16)))

    check_cost_matrices_against_scipy(matrices)


def test_munkres_min_32x32():
    matrices = []
    for i in range(0, 100):
        matrices.append(torch.randint(0, 100, (32, 32)))

    check_cost_matrices_against_scipy(matrices)


def test_munkres_min_64x64():
    matrices = []
    for i in range(0, 50):
        matrices.append(torch.randint(0, 100, (64, 64)))

    check_cost_matrices_against_scipy(matrices)


def test_munkres_min_128x128():
    matrices = []
    for i in range(0, 25):
        matrices.append(torch.randint(0, 100, (128, 128)))

    check_cost_matrices_against_scipy(matrices)


def check_cost_matrices_against_scipy(matrices: List[torch.Tensor]):
    for cost_matrix in matrices:
        correct_sum = calc_scipy_sum(cost_matrix.clone())

        lap = LinearAssignmentMunkres(cost_matrix)
        lap.fit()
        _, sum = lap.transform()

        assert sum == correct_sum


def calc_scipy_sum(cost_matrix: torch.Tensor):
    rows, cols = linear_sum_assignment(cost_matrix.numpy())
    sum = 0
    for f, s in zip(rows, cols):
        sum += cost_matrix[f, s]

    return sum
