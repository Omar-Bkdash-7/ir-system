class IRSystemException(Exception):
    """Base exception for IR System"""
    pass

class IndexNotFoundError(IRSystemException):
    """Raised when index is not found"""
    pass

class InvalidQueryError(IRSystemException):
    """Raised when query is invalid"""
    pass

class RetrievalError(IRSystemException):
    """Raised when retrieval fails"""
    pass

class RankingError(IRSystemException):
    """Raised when ranking fails"""
    pass

class PreprocessingError(IRSystemException):
    """Raised when preprocessing fails"""
    pass

class DatasetValidationError(IRSystemException):
    """Raised when dataset validation fails"""
    pass

class ForbiddenDatasetError(IRSystemException):
    """Raised when forbidden dataset is accessed"""
    pass
