"""Access control and the policies that depend on it.

See `access.py` for why withholding evidence is a decision point rather than a
filter step, and `injection.py` for treating retrieved text as hostile input.
"""

from app.security.access import (
    AccessDecision,
    AuthorisationOutcome,
    Principal,
    authorise,
    filter_candidates,
)
from app.security.injection import (
    InjectionFinding,
    InjectionReport,
    sanitize_for_context,
    scan,
)

__all__ = [
    "AccessDecision",
    "AuthorisationOutcome",
    "InjectionFinding",
    "InjectionReport",
    "Principal",
    "authorise",
    "filter_candidates",
    "sanitize_for_context",
    "scan",
]
