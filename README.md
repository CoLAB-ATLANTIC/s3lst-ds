LST_downscaling | Land Surface Temperature Downscaling
=============================
Algorithm to downscale Sentinel-3 LST data from 1 km to 300m.

## LST downscaling package includes:
- Image search and downloading by AOI
- Sentinel-3 product reading and transforming
- Statistical and ML downscaling model pipelines
- Model accuracy assesment pipeline


## Installation

1. Clone the [GitHub repository](https://github.com/CoLAB-ATLANTIC/LST_downscaling) to
   local directory `lst_downscaling` and change to it:

    ```bash
    git clone https://github.com/CoLAB-ATLANTIC/s3lst_downscale.git lst_downscaling
    cd lst_downscaling
    ```

2. Run script `scripts/install/install.sh` in the current shell:

    ```bash
    source scripts/install/install.sh
    ```

> [!NOTE]
> #### About script `scripts/install/install.sh`
> Script `scripts/install/install.sh` performs a clean installation of the project,
> uninstalling previous dependencies. It creates the [uv](https://docs.astral.sh/uv/)
> virtual environment `lst-downscaling` in folder `.venv` at the root directory and
> installs the dependencies defined in file [`pyproject.toml`](pyproject.toml) as well
> as the project files in editable state. The resultant package of project files is
> named `lst_downscaling`. By installing them, any project file may then be able to
> import objects from any other through the paths of these latter with respect to the
> project main directory.
>
> The script also sets [CDSE](https://dataspace.copernicus.eu/) credentials (stated in
> [`scripts/config.sh`](scripts/config.sh)) in the `~/.bashrc` file, removing old ones
> (to be able to download the Sentinel-3 data).
>
> #### About esa-snappy
>
> [esa-snappy](https://github.com/senbox-org/esa-snappy) Python package, together with
> its Java backend [SNAP](https://step.esa.int/main/download/snap-download/) may be used
> in place of virtual environment's
> [`rioxarray`](https://corteva.github.io/rioxarray/html/rioxarray.html) package to more
> accurately georeference the Sentinel-3 products. To install and configure SNAP and
> `esa-snappy` (and with the version stated in [`scripts/config.sh`](scripts/config.sh)), use
> flag `--install-snap` in the above-mentioned installation command,
> that is,
>
> ```bash
> source scripts/install/install.sh --install-snap
> ```
> 
> or, if the installation command was already run without the `--install-snap` flag, use
> the dedicated installation script `scripts/install/install_snap.sh` after, that is,
> 
> ```bash
> source scripts/install/install_snap.sh
> ```
> 
> Note that the Python version of `uv` virtual environment is relevant to `esa-snappy`:
> [this package might not be supported by the latest Python
> versions](https://senbox.atlassian.net/wiki/spaces/SNAP/pages/3114106881/Installation+and+configuration+of+the+SNAP-Python+esa_snappy+interface+SNAP+version+12).
>
> Furthermore, note that when importing `esa_snappy` package in Python a lot of errors
> and warnings may be printed. Still, [according to ESA's SNAP developers, these may be
> safely ignored](https://forum.step.esa.int/t/snap-gpt-warning/43343/4).
>
> Lastly, it is important to take into mind that `esa-snappy` uses Java in the
> background and limits the memory consumed by it. To increase Java's maximum heap
> memory to for instance 64 GB, one would need to open file `esa_snappy.ini` in the
> directory associated with `uv` virtual environment's `esa_snappy` package by doing 
>
> ```bash
> nano .venv/lib/python3.12/site-packages/esa_snappy/esa_snappy.ini
> ```
>
> and writing in it:
>
> ```ini
> [DEFAULT]
> java_max_mem: 64G
> ```

## Reinstall dependencies

Recreate the uv virtual environment `lst-downscaling` and reinstall dependencies through
the same script that was used in the installation:

```bash
source scripts/install/install.sh
```

To (re)install SNAP, use the `--install-snap` flag.

## Reconfigure

File [`scripts/config.sh`](scripts/config.sh) contains variables describing SNAP version
(`SNAP_VERSION`), and [CDSE](https://dataspace.copernicus.eu/) credentials (to be able
to download the Sentinel-3 data) (`CDSE_USER` and `CDSE_PASS`). The user may freely
change these. Note, however, to make the changes have an impact, one must [reinstall the
dependencies](#reinstall-dependencies).
