"""Custom exceptions raised by the weather impact ML core."""


class WeatherImpactError(Exception):
    """Base class for all errors raised by this project."""


class DataValidationError(WeatherImpactError):
    """A dataset does not meet the required schema or value rules."""


class InvalidInputError(WeatherImpactError):
    """User-supplied prediction or risk inputs are missing or out of range."""


class ModelNotFoundError(WeatherImpactError):
    """A trained model artifact could not be found on disk."""


class ScenarioError(WeatherImpactError):
    """A What-If scenario is invalid or produces unusable conditions."""


class WeatherServiceError(WeatherImpactError):
    """The live weather service is unreachable, returned bad data, or was given bad input."""
