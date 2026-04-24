"""Custom exception classes for the AI decision engine integrations."""


class GeminiAuthError(Exception):
    """Raised when Gemini authentication fails (no valid credentials found)."""
    pass


class GeminiInvalidChoice(Exception):
    """Raised when the AI returns an action that Showdown rejects.

    Used by both Gemini and Claude engines.
    """
    pass


class GeminiTimeout(Exception):
    """Raised when the AI API call exceeds the allowed time budget.

    Used by both Gemini and Claude engines.
    """
    pass


class GeminiUnsupportedFormat(Exception):
    """Raised when the detected format cannot be handled by the AI engine."""
    pass
