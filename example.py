from linear_assignment import LinearAssignmentMunkres
import torch

cost_matrix = torch.Tensor([[80, 23, 80, 39, 55, 2],
                           [19, 59, 36, 98, 19, 94],
                           [82, 38, 48, 95, 65, 25],
                           [9, 91, 83, 8, 48, 64],
                           [73, 22, 23, 86, 33, 42],
                           [2, 20, 9, 81, 94, 16]])

lap = LinearAssignmentMunkres(cost_matrix, maximize=False)
lap.fit()
indices, sum = lap.transform()
