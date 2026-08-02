import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm as tqdm_rich

# NOTE: `tqdm.rich` is still experimental, and raises warnings when used. These are
# suppressed here.
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# Set variable `tqdm` as Rich `tqdm` class
tqdm = tqdm_rich
