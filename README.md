
# Linear Assignment Problem (LAP)

The Linear Assignment Problem is a combinatorial optimization problem, where you want to find an optimial assignment between $n$ workers and $n$ tasks. Let $\{w_1, ..., w_n \}$ be the set of workers and let $\{ t_1, ..., t_n \}$ be the set of tasks. Let $c_{i,j}$ represent the cost of assigning $w_i \to t_j$ and let $\mathcal{\Alpha}$ be the set of sets of $n!$ permutations of $\{1, 2, ..., n\}$, for instance $\mathcal{\Alpha} = \{(1,2,...,n), (2,1,...,n), ..., (n, n-1, ..., 1) \}$ and so on. Then we can formulate LAP as

$$
\begin{align}

\alpha^* = \underset{\alpha \in \mathcal{\Alpha}}{\arg\min} \sum_{i=1}^n c_{i,\alpha(i)}



\end{align}
$$

where $\alpha^* \in \mathcal{\Alpha}$ is the optimal assignment, where $\alpha(i)_k$ represent the index of some task in $T$ assigned to worker $w_i$, where $k$ is the index of the permutation $\alpha_k \in \mathcal{\Alpha}$. It's also common to formulate it in terms of maximization, where let's say every worker produce a gain instead of a cost.

The motivation behind LAP is to find an algorithm that is significantly more efficient than $\mathcal{O}(n!)$, which would be the brute-force approach of testing every possible permutation of assignments.

For the different LAP algorithms, it is common to formulate them in terms of either matrices or bipartite graphs. 

## Munkres Algorithm

[(PAPER)](https://www.math.ucdavis.edu/~saito/data/emd/munkres.pdf)

The Munkres LAP Algorithm based on previous works from other computer scientist (See paper), is a pretty intuitive version of LAP, where we can directly apply it on a matrix of costs

$$
\begin{bmatrix}
c_{1,1} & c_{1,2} & \cdots & c_{1,n} \\
c_{2,1} & c_{2,2} & \cdots & c_{2,n} \\
\vdots  & \vdots  & \ddots & \vdots \\
c_{n,1} & c_{n,2} & \cdots & c_{n,n}
\end{bmatrix}
$$

following the above definitions. 

Though in practice, the Munkres Algorithm is very slow for larger cost matrices and not recommended to use.

## Future of this project
I may implement more efficient versions of this algorithm, starting with Jonker and Volgenant. 

