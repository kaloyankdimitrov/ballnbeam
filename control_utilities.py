import jax.numpy as jnp
def check_controllability(A, B):
    """
    Check if the system defined by (A, B) is controllable.
    
    Parameters:
    A (jnp.ndarray): State transition matrix.
    B (jnp.ndarray): Control input matrix.
    
    Returns:
    tuple: A boolean indicating if the system is controllable, and the rank of the controllability matrix.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrix B must have the same number of rows as A.")
    n = A.shape[0]
    
    # Build C = [B, AB, A^2B, ..., A^(n-1)B]
    controllability_blocks = [B]
    for k in range(1, n):
        controllability_blocks.append(jnp.linalg.matrix_power(A, k) @ B)
    C = jnp.hstack(controllability_blocks)
    
    rank_C = jnp.linalg.matrix_rank(C)
    
    return rank_C == n, rank_C