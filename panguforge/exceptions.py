"""panguforge exception hierarchy."""


class PanguError(Exception):
    """Base error for all panguforge failures."""


class ConfigError(PanguError):
    """Raised when configuration is missing or malformed."""


class BackendError(PanguError):
    """Raised when a compute backend fails or is unavailable."""


class DataError(PanguError):
    """Raised when a data pipeline stage fails."""


class TrainingError(PanguError):
    """Raised when a training run fails."""


class ServingError(PanguError):
    """Raised when the HTTP serving layer fails."""


class DistributedError(PanguError):
    """Raised when a distributed primitive fails."""
