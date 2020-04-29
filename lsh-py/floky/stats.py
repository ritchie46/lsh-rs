import numpy as np
from scipy import stats


def collision_prob_l2(r: float, distance: float) -> float:
    """
    Compute hash collision probability of L2

    Parameters
    ----------
    r : float
        Hyperparameter r
    distance : float
        Distance R

    Returns
    -------
    P1
    """
    # https://arxiv.org/pdf/1411.3787.pdf eq. 10
    a = 1 - 2 * stats.norm.cdf(-r / distance)
    b = (
        2
        / (np.sqrt(2 * np.pi) * r / distance)
        * (1 - np.exp(-(r ** 2) / (2 * distance ** 2)))
    )
    return a - b


def collision_prob_cosine(sim: float) -> float:
    """
    Compute hash collision probability of L2

    Parameters
    ----------
    sim : float
        Cosine similarity.

    Returns
    -------
    P1
    """
    return 1.0 - np.arccos(sim) / np.pi


def det_prob_query(p1: float, k: int, l: int) -> float:
    """
    Compute the probability of finding point q < cR

    Parameters
    ----------
    p1
        P_1 as determined with Union[floky.stats.collision_prob_cosine, floky.stats.collision_prob_l2]
    k
        Number of hash digits.
    l
        Number of hash tables.

    Returns
    -------
    Pq
        Prob. of finding point q < cR
    """
    return 1.0 - (1.0 - p1 ** k) ** l
