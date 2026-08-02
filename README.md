# **s3lst-ds**: a package for downscaling Sentinel-3 LST data using a scale-invariance-based model

<!-- Badges from [Shields.io](https://shields.io/badges) -->

<!-- ----------------------------- PyPI badges ----------------------------- -->
<!-- NOTE: the values of all PyPI badges are inferred from the PyPI website dedicated
to the project. All PyPI-related badges may be found
[here](https://shields.io/search/?q=pypi)-->
<!-- Packge version: https://shields.io/badges/py-pi-version -->
<!-- Pakage python version: https://shields.io/badges/py-pi-python-version -->
<!-- Package license: https://shields.io/badges/py-pi-license -->
<!-- Package implementation: https://shields.io/badges/py-pi-implementation -->
<!-- Package indicator for availability of wheel distribution : https://shields.io/badges/py-pi-wheel -->
<!-- Package development status: https://shields.io/badges/py-pi-status -->

<!-- ---------------------------- GitHub badges ---------------------------- -->
<!-- NOTE: the values of all GitHub badges are inferred from the GitHub repo of the
project. All GitHub-related badges may be found
[here](https://shields.io/search/?q=github)-->

<!-- GitHub time of last commit: https://shields.io/badges/git-hub-last-commit -->

<!-- ---------------------------- Other badges ----------------------------- -->
<!-- uv package and project manager usage: https://github.com/astral-sh/uv/pull/15075#issue-3291641128 -->
<!-- Ruff linter and formatter usage: https://github.com/astral-sh/ruff/blob/main/README.md?plain=1 -->
<!-- Hatch build backend usage: https://hatch.pypa.io/dev/next-steps/#community -->
![PyPI Version](https://img.shields.io/pypi/v/s3lst-ds)
![PyPI Python Version](https://img.shields.io/pypi/pyversions/s3lst-ds)
![PyPI License](https://img.shields.io/pypi/l/s3lst-ds)
![PyPI Implementation](https://img.shields.io/pypi/implementation/s3lst-ds)
![PyPI Wheel](https://img.shields.io/pypi/wheel/s3lst-ds)
![PyPI Status](https://img.shields.io/pypi/status/s3lst-ds)
![GitHub last commit](https://img.shields.io/github/last-commit/eliocp/s3lst-ds)
[![uv](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FOnyx-Nostalgia%2Fuv%2Frefs%2Fheads%2Ffix%2Flogo-badge%2Fassets%2Fbadge%2Fv0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Hatch project](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pypa/hatch/master/docs/assets/badge/v0.json)](https://github.com/pypa/hatch)

[s3lst-ds](https://github.com/CoLAB-ATLANTIC/s3lst-ds) provides pipelines for
conveniently querying, downloading, filtering and downscaling [Sentinel-3
LST](https://sentiwiki.copernicus.eu/web/slstr-products#L2-LST-Products) data using
either single or multi-timestamp scale-invariance-based models.


## (*Base*) Installation

* Install the latest stable release from [PyPI](https://pypi.org/project/s3lst-ds/) in
the activated virtual environment using [`pip`](https://pypi.org/project/pip/) command:

    ```bash
    pip install s3lst-ds
    ```
    or [`uv`](https://docs.astral.sh/uv/):

    ```bash
    uv add s3lst-ds
    ```

### Set CDSE credentials

[CDSE](https://dataspace.copernicus.eu/) credentials are required to download the
Sentinel-3 data and must be safely set as environment variables in the system. This may
be done through an appropriate script, either using the respective executable:

```bash
VENV/bin/s3lst-ds-set-cdse
```

(where `VENV` is the path to the installation directory of the virtual environment (e.g.
`.venv`)), or, more conveniently, using `uv`:

```bash
uv run s3lst-ds-set-cdse
```

> [!NOTE]
> The credentials will be written to file `~/.config/cdse_credentials.sh` with read and
> write permissions solely issued to the user.
>
> Note that if the user would like to remove the credentials at a later time, an appropriate script could be run through the respective excutable:
> ```bash
> VENV/bin/s3lst-ds-unset-cdse
> ```
> or, using `uv`:
> 
> ```bash
> uv run s3lst-ds-unset-cdse
> ```

## (*Optional*) Snappy Installation

The base installation considers
[`rioxarray`](https://corteva.github.io/rioxarray/stable/) for georeferencing the
Sentinel-3 products. However, [`snappy`](https://github.com/senbox-org/esa-snappy) has
been found to produce better results, and, because of that, it is herein availed as an
optional tool. The installation of this extra dependency requires two steps, in the
following order:


1. Installation of the `snappy` Python package using `pip`:

    ```bash
    pip install s3lst-ds[snap]
    ```

    or `uv`:

    ```bash
    uv add s3lst-ds[snap]
    ```

2. Installation of the backend [`SNAP`](https://earth.esa.int/eogateway/tools/snap) Java
   package and subsequent configuration using the respective executable:

    ```bash
    VENV/bin/s3lst-ds-install-snap
    ```

    or, more conveniently, using `uv`:

    ```bash
    uv run s3lst-ds-install-snap
    ```

> [!WARNING]
> It is important to note that SNAP is configured with a limited amount of memory. To
> increase SNAP's maximum heap memory to for instance 64 GB, one would need to open file
> `esa_snappy.ini` nested in the installation directory of the virtual environment
> (herein assumed to be `VENV`) by doing 
>
> ```bash
> nano VENV/lib/python3.12/site-packages/esa_snappy/esa_snappy.ini
> ```
> and writing in it:
>
> ```ini
> [DEFAULT]
> java_max_mem: 64G
> ```

> [!NOTE]
> If the user would to like to uninstall `SNAP` at a later time, an appropriate script
> may be used, either through the respective executable:
>
> ```bash
> VENV/bin/s3lst-ds-uninstall-snap
> ```
> or, more conveniently, using `uv`:
> 
> ```bash
> uv run s3lst-ds-uninstall-snap
> ```

## Documentation

To be built.

## Development

### Installation of development dependencies

* Create the development environment using all dependency groups:

    ```bash
    uv sync --all-groups
    ```

### Linting

* Use [`ruff`](https://docs.astral.sh/ruff/linter/) for Python linting:

    ```bash
    uv run ruff check --fix --diff
    ```
    The command above will check for bugs, suspicious code, style violations, dead code,
    complexity issues and import problems. It will further proposed fixes.

* Apply the fixes:

    ```bash
    uv run ruff check --fix
    ```

### Formatting

* Use [`ruff`](https://docs.astral.sh/ruff/formatter/) for Python formatting:

    ```bash
    uv run ruff format --diff
    ```

    The command above will check the code structure, namely, the indentation, spaces,
    line breaks, quote style and long line wrapping. It will further show the proposed
    changes.

* Apply the changes:

    ```bash
    uv run ruff format
    ```

## License

Pingi is licensed under the terms of the [MIT
license](https://github.com/eliocp/pingi/blob/main/LICENSE). 