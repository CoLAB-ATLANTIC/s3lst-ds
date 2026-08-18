# Releasing

This document is intended for project maintainers
([@armkhudinyan](https://github.com/armkhudinyan) and
[@eliocp](https://github.com/eliocp)). The document describes how, after successful
merge pull requests, the maintainer may:

- build the package and publish it to [PyPI](https://pypi.org/).
- create [GitHub
  releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
- deploy the documentation to [GitHub Pages](https://docs.github.com/en/pages).

## Requirements

To publish the package to PyPI and create a GitHub release, you would need:

- [`uv`](https://docs.astral.sh/uv/) project manager.
- [`git`](https://git-scm.com/) version control system.
- Repo owner [`CoLAB-ATLANTIC`](https://github.com/CoLAB-ATLANTIC) set as a
  [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/#github-actions)
  of the [PyPI project](https://pypi.org/project/s3lst-ds/).

## Release Workflow

### Approve pull request

- Review a pull request into main, provide comments, request changes if necessary, and
  approve it when appropriate.

> [!NOTE]
>
> #### Automatic executions after approval of a pull request
>
> As defined in the GitHub workflow file
> [`ci.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/ci.yml),
> several checks are executed by GitHub after a pull request being approved:
>
> - Code lint check (using [`ruff check`](https://docs.astral.sh/ruff/linter/)).
> - Code format check (using
>   [`ruff format --check`](https://docs.astral.sh/ruff/formatter/)).
>
> Note that although the workflow uses `push` to in the mapping of the trigger, approval
> of pull requests are also included since these always involve a push. Also note that
> regardless of any fail in workflow, the completion of the pull request proceeds.

### Update project version

- Switch to the main branch, pull the changes associated with the pull request and
  update the version of the project stated in
  [`pyproject.toml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/pyproject.toml):

    ```bash
    git switch main
    git pull
    uv version VERSION
    ```

    where `VERSION` is the version.

- Commit and push this change:

    ```bash
    git add -A
    git commit -m "Updated project version."
    git push
    ```

### Create release tag

- Associate tag `VERSION` (the version considered in the previous step) with the latest
  commit and push the tag to the remote repo:

    ```bash
    git tag VERSION
    git push origin VERSION
    ```

> [!NOTE]
>
> #### Semantic versioning
>
> `VERSION` must satisfy the [semantic versioning](https://semver.org/) rules as
> described in the table below.
>
> | Release Type      | Example        |
> | ----------------- | -------------- |
> | Major release     | v1.0.0         |
> | Minor release     | v1.1.0         |
> | Patch release     | v1.1.1         |
> | Alpha pre-release | v1.1.1-alpha.1 |
> | Beta pre-release  | v1.1.1-beta.1  |
> | Release candidate | v1.1.1.-rc.1   |

### Let GitHub Actions automatically build and publish the package to PyPI, and deploy the documentation to GitHub Pages

- After pushing the tag, the package is automatically built and published to
  [PyPI](https://pypi.org/) by GitHub using workflow file
  [`build-publish-package.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/build-publish-package.yml).
  The published new version of the package would then appear in the project
  [`s3lst-ds`](https://pypi.org/project/s3lst-ds/) hosted in the PyPI website. Moreover,
  the documentation would also be published to [GitHub
  Pages](https://docs.github.com/en/pages) using workflow file
  [`publish-docs.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/publish-docs.yml).
  
> [!NOTE]
> The Continuous Integration workflow file
> [`ci.yml`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/blob/main/.github/workflows/ci.yml)
> would identically run since it is triggered by any kind of push.

### (Optional) Create GitHub release

- If desired, create a
  [GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
  by going to the GitHub page, clicking on
  [`Releases`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/releases), then
  [`Draft a new release`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/releases/new) and
  issuing a title (usually the project version `VERSION`) and some notes. After this,
  the GitHub release would then appear in the
  [`Releases`](https://github.com/CoLAB-ATLANTIC/s3lst-ds/releases) section of the repo
  GitHub page.

> [!NOTE]
>
> #### GitHub release
>
> GitHub releases are portable "snapshots" of a specific version (commit) of the project
> containing release notes and source archives (`zip` and `tar.gz` files) which may be
> used in other contexts outside of GitHub.
