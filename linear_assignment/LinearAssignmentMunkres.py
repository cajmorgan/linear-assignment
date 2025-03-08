import torch
from torch import Tensor


class LinearAssignmentMunkres():

    def __init__(self, cost_matrix: Tensor, maximize=False, device="cpu"):
        """
        Args:
            cost_matrix (Tensor): A nxn matrix .
            maximize (bool):  Whether to maximize the assignment (or not)
            device (str, optional): The PyTorch device to be used, defaults to "cpu".
        """

        self.org_cost_matrix = cost_matrix.clone()
        if maximize:
            cost_matrix = cost_matrix.max() - cost_matrix

        self.n_dim = cost_matrix.shape[0]
        self.cost_matrix = cost_matrix
        self.covered = torch.zeros_like(cost_matrix)
        self.zeros = (cost_matrix == 0).int()
        self.starred_zeros = self.zeros.clone()

        self.starred_zeros, self.covered = self.__star_and_cover_col_independent_zeros(
            self.starred_zeros, self.covered)

        self.non_covered_zeros_mask = self.zeros - self.covered*self.zeros == 1
        self.non_covered_zeros_indices = torch.nonzero(
            ~self.covered.bool() & self.zeros)

        self.primed_zeros = torch.zeros_like(cost_matrix)
        self.finished = False

        # Move tensors to device
        self.cost_matrix = self.cost_matrix.to(device)
        self.starred_zeros = self.starred_zeros.to(device)
        self.covered = self.covered.to(device)
        self.primed_zeros = self.primed_zeros.to(device)
        self.zeros = self.zeros.to(device)

    def __star_and_cover_col_independent_zeros(self, starred_zeros: Tensor, covered: Tensor):
        # star independent zeros
        starred_zeros, covered = starred_zeros.clone(), covered.clone()
        starred_zeros_indices = torch.nonzero(starred_zeros)

        for zero_idx in starred_zeros_indices:
            row_idx, col_idx = zero_idx
            # Skip if this row/col has a starred zero
            if not starred_zeros[row_idx, col_idx]:
                continue

            starred_zeros[row_idx, :] = 0
            starred_zeros[:, col_idx] = 0
            starred_zeros[row_idx, col_idx] = 1
            covered[:, col_idx] += 1

        return starred_zeros, covered

    def fit(self):
        cover_changed = False
        while not self.finished:
            # While non_covered_zeros_indices holds elements, we loop
            while self.non_covered_zeros_indices.numel() > 0:
                # Step 1
                zero_idx = self.non_covered_zeros_indices[0]
                row_idx, col_idx = zero_idx
                self.primed_zeros[row_idx, col_idx] = 1

                # # Check if there is a starred zero in row_idx
                if self.starred_zeros[row_idx, :].sum() == 0:
                    cover_changed = True
                    # goto step 2
                    # This step is mainly for finding another unique set of primed zeros,
                    # by unstaring the previous starred zeros found in the sequence
                    # and starring the primed zeros in the sequence
                    # Step 2 will allow us to always cover 1 more zero than previously

                    current_zero_row_idx, current_zero_col_idx = (
                        row_idx, col_idx)
                    sequence = [(current_zero_row_idx, current_zero_col_idx)]
                    counter = 0

                    while not (counter % 2 == 0 and self.starred_zeros[:, current_zero_col_idx].sum() == 0):

                        # Initialize
                        next_zero_row_idx = 0
                        next_zero_col_idx = 0

                        if counter % 2 == 0:
                            # find starred zero in col of current zero
                            next_zero_row_idx = torch.argmax(
                                self.starred_zeros[:, current_zero_col_idx])
                            next_zero_col_idx = current_zero_col_idx

                        else:
                            # find primed zero in row of current zero
                            next_zero_col_idx = torch.argmax(
                                self.primed_zeros[current_zero_row_idx, :])
                            next_zero_row_idx = current_zero_row_idx

                        current_zero_row_idx, current_zero_col_idx = next_zero_row_idx, next_zero_col_idx
                        sequence.append(
                            (current_zero_row_idx, current_zero_col_idx))
                        counter += 1

                    for seq_idx, (zero_row_idx, zero_col_idx) in enumerate(sequence):
                        if seq_idx % 2 == 0:
                            self.starred_zeros[zero_row_idx, zero_col_idx] = 1
                        else:
                            self.starred_zeros[zero_row_idx, zero_col_idx] = 0

                    # Erase primes
                    self.primed_zeros = self.primed_zeros.zero_()
                    # Uncover all covered rows
                    covered_row_indices = self.covered.bool().sum(dim=1) == self.n_dim
                    self.covered[covered_row_indices, :] -= 1

                    starred_zeros_col_indices = self.starred_zeros.sum(
                        dim=0).bool()
                    self.covered[:, starred_zeros_col_indices] += 1
                    self.covered = self.covered.bool().int()

                    # Cover independent zeros
                    if self.starred_zeros.sum() == self.n_dim:
                        finished = True
                        break

                    self.non_covered_zeros_mask = (
                        self.zeros == 1) & (self.covered == 0)
                    self.non_covered_zeros_indices = torch.nonzero(
                        self.non_covered_zeros_mask)
                    continue

                starred_zero_in_row_col_idx = torch.argmax(
                    self.starred_zeros[row_idx, :])
                self.covered[row_idx, :] += 1
                self.covered[:, starred_zero_in_row_col_idx] -= 1

                self.non_covered_zeros_mask = (
                    self.zeros == 1) & (self.covered == 0)
                self.non_covered_zeros_indices = torch.nonzero(
                    self.non_covered_zeros_mask)

            if self.finished or self.covered.all():
                break

            # step 3
            h = self.cost_matrix[~self.covered.bool()].min()

            covered_row_indices = self.covered.bool().sum(dim=1) == self.n_dim
            ucovered_col_indices = self.covered.bool().sum(dim=0) < self.n_dim

            self.cost_matrix[covered_row_indices, :] += h
            self.cost_matrix[:, ucovered_col_indices] -= h

            # Reset zeros and update non_covered_zeros_indices
            self.zeros = (self.cost_matrix == 0).int()
            self.non_covered_zeros_mask = (
                self.zeros == 1) & (self.covered == 0)
            self.non_covered_zeros_indices = torch.nonzero(
                self.non_covered_zeros_mask)

    def transform(self) -> tuple[Tensor, Tensor | float]:
        starred_zeros_indices = torch.nonzero(self.starred_zeros)
        sum = 0
        for row, col in starred_zeros_indices:
            sum += self.org_cost_matrix[row, col].item()

        return (starred_zeros_indices, sum)
