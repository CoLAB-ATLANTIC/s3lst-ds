class ConfigError(Exception):
    pass


class FailedProdReadingError(Exception):
    pass


class UnzipError(Exception):
    pass


class NoProductFoundError(Exception):
    pass


class NoMatchingProductError(Exception):
    pass


class MultipleMatchingProdutsError(Exception):
    pass


class UnsupportedProductError(Exception):
    pass


class InvalidProductError(Exception):
    pass


class ReprojectionError(Exception):
    pass


class CleanupError(Exception):
    pass


class CloudCoverComputationError(Exception):
    pass


class CloudCoverLimitError(Exception):
    pass


class DataWranglingError(Exception):
    pass


class DataBatchingError(Exception):
    pass


class MaskCloudsError(Exception):
    pass


class TrainingError(Exception):
    pass


class TuningError(Exception):
    pass


class DownscalingError(Exception):
    pass


class ScoringError(Exception):
    pass


class ReadingError(Exception):
    pass


class WritingError(Exception):
    pass


class AccessTokenGenerationError(Exception):
    pass


class AccessTokenRefreshError(Exception):
    pass


class JSONDecodeError(Exception):
    pass


class GeometryError(Exception):
    pass
