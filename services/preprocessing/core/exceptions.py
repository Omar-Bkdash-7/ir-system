from common.exceptions import (
    PreprocessingError,
)


class ResourceNotFoundError(PreprocessingError):
    pass


class TokenizationError(PreprocessingError):
    pass
