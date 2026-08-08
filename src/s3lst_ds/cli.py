import logging
import stat
import subprocess
from importlib.resources import files
from pathlib import Path

from rich.prompt import Prompt

from s3lst_ds.utilities.logging_utils import RichLogger


def set_cdse() -> None:
    """
    Set CDSE credentials to the configuration file `~/.config/cdse_credentials.sh`
    and source it in the `~/.bashrc` so that the credentials may be regarded as
    environment variables. This is convenient for downloading Sentinel-3 data using the
    OData API. For the sake of security, give permissions for reading and writing of the
    configuration file solely to the user.
    """

    # ---> Create logger
    logger = RichLogger(
        name="s3lst_ds.cli.set_cdse",
        level=logging.INFO,
        log_mode="console",
    )

    logger.info("Setting CDSE credentials...")

    # ---> Get CDSE credentials, write them to the configuration file and give permissions for reading and writing solely to the user
    # Prompt user for CDSE credentials
    cdse_user = Prompt.ask(prompt="         CDSE email")
    cdse_pass = Prompt.ask(prompt="         CDSE password", password=True)
    # Define the path to the configuration file
    path_config = Path.home() / ".config" / "cdse_credentials.sh"
    # Create the parent directory if it doesn't exist
    path_config.parent.mkdir(parents=True, exist_ok=True)
    # Create file and write the credentials to it, overwriting if the file already
    # exists
    path_config.write_text(
        f"export CDSE_USER={cdse_user}\nexport CDSE_PASS={cdse_pass}"
    )
    # Make the configuration file readable/writable only to the user
    path_config.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # ---> Add command to source the configuration file to the bashrc file
    # Define command to source the configuration file
    cmd_source_config = f"source {path_config}"

    # Get path to bashrc file
    path_bashrc = Path.home() / ".bashrc"

    # Read the content of the bashrc file if it exists
    text_bashrc = (
        path_bashrc.read_text(encoding="utf-8") if path_bashrc.exists() else None
    )

    # Write the command to source the configuration file to the bashrc file if it
    # does not already exist in the latter
    if text_bashrc is None or (
        text_bashrc is not None and cmd_source_config not in text_bashrc
    ):
        with path_bashrc.open("a", encoding="utf-8") as f:
            f.write(
                (
                    "\n"
                    if text_bashrc is not None and not text_bashrc.endswith("\n")
                    else ""
                )
                + f"\n# Create environment variables for CDSE credentials"
                f"\n{cmd_source_config}"
                "\n"
            )

    logger.info("Done.")


def unset_cdse() -> None:
    """
    Remove CDSE credentials from configuration file `~/.config/cdse_credentials.sh`.
    Note that the respective environment variables would still be available in the
    current shell session until the user opens a new one.
    """

    # ---> Create logger
    logger = RichLogger(
        name="s3lst_ds.cli.unset_cdse",
        level=logging.INFO,
        log_mode="console",
    )

    logger.info("Removing CDSE credentials...")

    # Get the path to the configuration file
    path_config = Path.home() / ".config" / "cdse_credentials.sh"

    # Make the configuration file empty if it exists and is not already empty
    if not path_config.exists() or path_config.stat().st_size == 0:
        logger.info("Nothing done. CDSE credentials had not been set.")
    else:
        # Overwrite file content with empty string
        path_config.write_text("")

        logger.info("Done.")


def install_snap() -> None:
    """
    (Re)install SNAP dependencies and configure snappy by running the installation
    script `scripts/install/install_snap.sh`.
    """
    # Path to latex installation script
    script = files("s3lst_ds").joinpath("scripts/install/install_snap.sh")
    # Run script
    subprocess.run(
        # Command arguments
        ["bash", str(script)],
        # Raise error if the command fails
        check=True,
    )


def uninstall_snap() -> None:
    """
    Uninstall SNAP dependencies by running the uninstallation script
    `scripts/install/uninstall_snap.sh`.
    """
    # Path to SNAP uninstallation script
    script = files("s3lst_ds").joinpath("scripts/install/uninstall_snap.sh")
    # Run script
    subprocess.run(
        # Command arguments
        ["bash", str(script)],
        # Raise error if the command fails
        check=True,
    )
