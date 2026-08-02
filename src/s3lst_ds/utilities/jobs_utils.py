import os


def parse_n_jobs(n_jobs: int | None) -> int:
    """
    Parse number of jobs `n_jobs` in similarity to what `joblib` does:
        - if `n_jobs = None`: use `1` processor;
        - if `n_jobs > 0`: use exactly `n_jobs` processors;
        - if `n_jobs = -k` with `k > 0`: use all processors except `k-1`;

    For more details see
    https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    Parameters
    ----------
    n_jobs : int or None
        The encoded number of processors to regard. If `None`, it is regarded as `1`.

    Returns
    -------
    n_jobs_parsed : int
        The actual number of processors to regard.

    """

    n_jobs = (
        1
        if n_jobs is None
        else max(1, n_jobs if n_jobs >= 0 else os.cpu_count() + n_jobs + 1)  # type: ignore
    )

    return n_jobs
