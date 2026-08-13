# Contributing

This document is intended for project developers. The document describes the development
workflow.

## Requirements

To develop the present project you would need:

- An [Unix](https://en.wikipedia.org/wiki/Unix)-like environment.
- [`uv`](https://docs.astral.sh/uv/) project manager.
- [`git`](https://git-scm.com/) version control system.
- A [CDSE](https://dataspace.copernicus.eu/) account.

## Contribution Workflow

### 1. Clone

- Clone the [GitHub repo](https://github.com/CoLAB-ATLANTIC/s3lst-ds):

  ```bash
  git clone git@github.com:CoLAB-ATLANTIC/s3lst-ds.git
  cd s3lst-ds
  ```

### 2. Branch

- Create a feature branch for your changes from
  [`main`](https://github.com/CoLAB-ATLANTIC/s3lst-ds):

  ```bash
  git switch main
  git pull
  git switch -c FEATURE/MY_CHANGE
  ```

  where `FEATURE/MY_CHANGE` is the name of the branch.

> [!TIP]
>
> #### Examples of development names
>
> - `feature/add-download-endpoint`
> - `fix/rasterio-import-error`
> - `docs/update-contributing`
> - `refactor/download-service`
> - `perf/parallel-downloads`
> - `test/add-download-tests`

### 3. Install dependencies

- Create projects's python environment with all dependency groups including the optional
  ones in accordance with
  [`pyproject.toml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/pyproject.toml):

  ```bash
  uv sync --all-groups --all-extras
  ```

- Install the backend [`SNAP`](https://earth.esa.int/eogateway/tools/snap) Java package
  (if not already installed) and subsequently configure it using `uv`:

  ```bash
  uv run s3lst-ds-install-snap
  ```

- Install all [git hooks](https://pre-commit.com/#supported-git-hooks) in accordance
  with
  [`.pre-commit-config.yaml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.pre-commit-config.yaml):

  ```bash
  uv run pre-commit install
  ```

- Safely set your [CDSE](https://dataspace.copernicus.eu/) credentials as environment
  variables of the system:

  ```bash
  uv run s3lst-ds-set-cdse
  ```

> [!NOTE]
>
> #### CDSE credentials
>
> CDSE credentials are required to download Sentinel-3 data. Script `s3lst-ds-set-cdse`
> will prompt the user to provide their CDSE mail and password. The script will
> subsequently write them to file `~/.config/cdse_credentials.sh` with user-only read
> and write permissions and source it in `~/.bashrc` file. You may check the created
> credentials file using the command:
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

### 4. Make changes

- Make your changes in the code by following the coding style guide of Python
  Enhancement Proposal 8 ([PEP 8](https://peps.python.org/pep-0008/)).

### 5. Commit and push changes

- Commit your changes and push them to the remote repo:

  ```bash
  git add -A
  git commit -m "MESSAGE"
  git push -u origin FEATURE/MY_CHANGE
  ```

  where `MESSAGE` is the commit message to set.

> [!NOTE]
>
> #### Automatic executions on commit
>
> As defined in
> [`.pre-commit-config.yaml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.pre-commit-config.yaml),
> [`pre-commit`](https://pre-commit.com/) package is in this project used to run several
> git hooks on commit:
>
> - Project syncing (using
>   [`uv sync`](https://docs.astral.sh/uv/reference/cli/#uv-sync)).
> - [`uv.lock`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/uv.lock) file
>   update (using [`uv lock`](https://docs.astral.sh/uv/reference/cli/#uv-lock)).
> - Code lint check (using [`ruff check`](https://docs.astral.sh/ruff/linter/)).
> - Code formatting (using [`ruff format`](https://docs.astral.sh/ruff/formatter/)).
>
> If any of these fails, the commit operation is aborted. Note that project syncing is
> also automatically performed after checkouts, merging, amending and rebasing.
>
> #### Automatic executions after a push
>
> As defined in the GitHub workflow file
> [`ci.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/ci.yml),
> several are executed by GitHub after a push:
>
> - Code lint check (using [`ruff check`](https://docs.astral.sh/ruff/linter/)).
> - Code format check (using
>   [`ruff format --check`](https://docs.astral.sh/ruff/formatter/)).
>
> Note that regardless of any of these failing, the push proceeds.

### 6. Open pull request

- In GitHub open a [pull request](https://github.com/CoLAB-ATLANTIC/s3lst-ds/compare)
  using `main` branch as base and the development branch as compare one, so that other
  developers and maintainers may review the changes, comment on and approve them and
  also request corrections.

> [!NOTE]
>
> #### Automatic executions on the creation of a pull request
>
> As defined in the GitHub workflow file
> [`ci.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/ci.yml),
> the push-specific jobs are also run on the creation of a pull request.
