"""Immutable configuration and environment parsing.

Every tunable in the system is a frozen dataclass field with a documented
default and an environment override. Frozen because a retrieval parameter
mutated mid-request makes a regression unreproducible: two runs of the same
query would differ for a reason that is not in the report."""
