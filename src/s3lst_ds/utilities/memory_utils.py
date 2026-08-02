import sys

# If the system is Linux or macOS, import resource package to limit memory usage
# NOTE: the resource package is only available in Linux or macOS
if sys.platform in ["linux", "darwin"]:
    import resource


def set_memory_limits(
    max_mem_soft: int | None = None, max_mem_hard: int | None = None
) -> None:
    """
    Set memory (RAM + swap) soft and hard limits to `max_mem_soft` and `max_mem_hard`,
    respectively.

    Note that the "soft limit" is termed in such a way because the process being limited
    can generally raise the current limit at will. The hard limit is the maximum value
    to which a process is allowed to set the soft limit. For more details, read this
    [Stack Overflow answer](https://stackoverflow.com/a/60411718/4382986).

    If the hard limit is surpassed, a `MemoryError` is thrown and script stops running.
    Note that memory limits can only be imposed if the system is Linux or macOS. To know
    more about `resource.setrlimit` read [method's official
    documentation](https://docs.python.org/3/library/resource.html#resource.setrlimit).

    Parameters
    ----------
    max_mem_soft : int or None
        Soft limit on consumed memory (GB). If `None`, it is set to `max_mem_hard`. If
        both are `None`, no memory limit is imposed.
    max_mem_hard : int or None
        Hard limit on consumed memory (GB). If `None`, it is set to `max_mem_soft`. If
        both are `None`, no memory limit is imposed.
    """
    if (
        sys.platform in ["linux", "darwin"]
        and max_mem_soft is not None
        or max_mem_hard is not None
    ):
        resource.setrlimit(
            # Total resource available
            # NOTE: resource.RLIMIT_AS refers to the total virtual memory (maximum
            # addressable memory, that is RAM + swap) in bytes [B].
            resource.RLIMIT_AS,  # type: ignore
            (
                # Soft limit in bytes [B] (the limit a process cannot exceed under
                # normal conditions)
                # NOTE: 1 GB = 1024**3 bytes
                (max_mem_soft if max_mem_soft is not None else max_mem_hard) * 1024**3,  # type: ignore
                # Hard limit in bytes [B] (the absolute maximum limit that the soft
                # limit can be increased to)
                (max_mem_hard if max_mem_hard is not None else max_mem_soft) * 1024**3,  # type: ignore
            ),
        )
