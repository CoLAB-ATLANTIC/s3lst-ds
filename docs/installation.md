# Installation

## Requirements

To be able to install and use the [s3lst-ds](https://github.com/CoLAB-ATLANTIC/s3lst-ds)
package in your project, you would need:

- An [Unix](https://en.wikipedia.org/wiki/Unix)-like environment.
- [`uv`](https://docs.astral.sh/uv/) project manager.
- A [CDSE](https://dataspace.copernicus.eu/) account.

## Installation Steps

### Install package from PyPI

- Install the latest stable release from [PyPI](https://pypi.org/project/s3lst-ds/) in
  the activated virtual environment using [`uv`](https://docs.astral.sh/uv/):

    ```bash
    uv add s3lst-ds
    ```

### Set CDSE credentials

- Safely set your [CDSE](https://dataspace.copernicus.eu/) credentials as environment
  variables of the system:

    ```bash
    uv run s3lst-ds-set-cdse
    ```

### Install [`esa-snappy`](https://github.com/senbox-org/esa-snappy) (optional)

- Install [`esa-snappy`](https://github.com/senbox-org/esa-snappy) Python package using
  `uv`:

    ```bash
    uv add s3lst-ds[snap]
    ```

- Install the backend [`SNAP`](https://earth.esa.int/eogateway/tools/snap) Java package
  (if not already installed) and subsequently configure it using `uv`:

    ```bash
    uv run s3lst-ds-install-snap
    ```

> [!NOTE]
>
> #### CDSE credentials
>
> CDSE credentials are required to download Sentinel-3 data. Script `s3lst-ds-set-cdse`
> will prompt you to provide their CDSE mail and password. The script will subsequently
> write them to file `~/.config/cdse_credentials.sh` with user-only read and write
> permissions and source it in `~/.bashrc` file. You may check the created credentials
> file using the command:
>
> ```bash
> nano ~/.config/cdse_credentials.sh
> ```
>
> Note that if you would like to remove the credentials from the file at a later time,
> you may run:
>
> ```bash
> uv run s3lst-ds-unset-cdse
> ```
>
> #### Better downscaling results may be obtained with `esa-snappy`
>
> The default installation considers
> [`rioxarray`](https://corteva.github.io/rioxarray/stable/) for georeferencing the
> downloaded Sentinel-3 products. However,
> [`esa-snappy`](https://github.com/senbox-org/esa-snappy) has been found to produce
> better results, and, because of that, it is herein availed as an optional tool. With
> its installation, both tools can be used for georeferencing.
>
> #### Uninstall `SNAP`
>
> If the you would to like to uninstall `SNAP` at a later time, you may run:
>
> ```bash
> uv run s3lst-ds-uninstall-snap
> ```
>
> #### `SNAP`'s memory limit
>
> It is important to note that `SNAP` is configured with a limited amount of memory. To
> increase `SNAP`'s maximum heap memory to for instance `64 GB`, you would need to open
> file `esa_snappy.ini` nested in the installation directory of the virtual environment
> (herein assumed to be `VENV`) by doing
>
> ```bash
> nano VENV/lib/python3.12/site-packages/esa_snappy/esa_snappy.ini
> ```
>
> and writing in it:
>
> ```ini
> [DEFAULT]
> java_max_mem: 64G
> ```
