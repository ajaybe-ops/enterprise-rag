"""Authorisation, and what to do when authorisation removes the answer.

The central decision
--------------------
Suppose the best-matching document for a query is a capacity allocation the
caller is not authorised to read. A post-filter removes it. The second-best
document is a public SLO that shares most of the vocabulary. A naive pipeline
now answers from the SLO, with a confident tone and a citation to a document the
user is allowed to see.

That is the worst outcome available. It is a policy violation *and* a silent
factual substitution, and the citation makes it look well-sourced. The user
cannot tell the system withheld something from them; they believe it answered.

So filtering is not the end of the decision. When authorisation removes evidence
that the retriever had actually ranked highly, this module downgrades the
response to a refusal. The caller learns that the answer exists and is
restricted, which is the only truthful option available, and the alternative --
quietly answering from a weaker source -- is removed from the code path
entirely.

The asymmetry is deliberate. A false refusal is an annoyance; a confidently wrong
answer derived from a document the user was not cleared for is a breach that
survives review, because the citation is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "AccessDecision",
    "Principal",
    "authorise",
    "filter_candidates",
    "AuthorisationOutcome",
]


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller.

    `tenants` is a set because enterprise access is rarely one-to-one: an
    operator routinely holds two tenants, and a service account legitimately
    spans several. Modelling it as a single `tenant` string makes the
    multi-tenant case the special case, and special cases in authorisation code
    are how leaks happen.
    """

    subject: str
    tenants: frozenset[str] = field(default_factory=lambda: frozenset({"public"}))
    roles: frozenset[str] = field(default_factory=frozenset)
    authenticated: bool = True

    @classmethod
    def anonymous(cls) -> "Principal":
        return cls(subject="anonymous", tenants=frozenset(), roles=frozenset(), authenticated=False)

    def may_read(self, tenant: str | None) -> bool:
        """Whether this principal may read content owned by `tenant`.

        Fails closed on every ambiguous input: no tenant on the document, no
        tenants on the principal, or an unauthenticated principal all resolve to
        "no". An allow-list that defaults to allow is an allow-list that will
        eventually leak.
        """
        if not self.authenticated:
            return False
        owner = (tenant or "").strip().lower()
        if not owner:
            return False
        if owner == "public":
            return True
        return owner in self.tenants

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "tenants": sorted(self.tenants),
            "roles": sorted(self.roles),
            "authenticated": self.authenticated,
        }


@dataclass(frozen=True, slots=True)
class AccessDecision:
    """The outcome of applying authorisation to one candidate set."""

    allowed: tuple = ()
    withheld: tuple = ()
    withheld_sources: tuple[str, ...] = ()
    withheld_count: int = 0

    @property
    def withheld_anything(self) -> bool:
        return bool(self.withheld)

    def to_dict(self) -> dict:
        return {
            "withheld_count": self.withheld_count,
            "withheld_sources": list(self.withheld_sources),
        }


def filter_candidates(
    candidates: Sequence,
    principal: Principal,
    *,
    tenant_key: str = "tenant",
) -> AccessDecision:
    """Split ranked candidates into readable and withheld.

    Preserves the original ranking order within each partition. Re-sorting after
    filtering would be a subtle way to change retrieval results as a side effect
    of a security check, which makes the security check impossible to review in
    isolation from the ranking logic.
    """
    allowed, withheld = [], []
    for candidate in candidates:
        metadata: Mapping = getattr(candidate, "metadata", {}) or {}
        owner = metadata.get(tenant_key)
        if principal.may_read(owner):
            allowed.append(candidate)
        else:
            withheld.append(candidate)

    return AccessDecision(
        allowed=tuple(allowed),
        withheld=tuple(withheld),
        withheld_sources=tuple(
            dict.fromkeys(getattr(c, "source", "?") for c in withheld)
        ),
        withheld_count=len(withheld),
    )


@dataclass(frozen=True, slots=True)
class AuthorisationOutcome:
    """What the answer layer should do after authorisation.

    `evidence_withheld_at_rank` is the load-bearing field. It is the rank of the
    best candidate that authorisation removed, and it distinguishes two
    situations that look identical from the outside:

        * the best match was restricted -> refuse, and say why
        * some low-ranked noise was restricted, and readable evidence remains
          at the top -> answer normally

    Without the rank, any withholding at all would trigger a refusal, and a
    system with a single restricted document in a large corpus would refuse most
    questions. The rank is what makes the policy proportionate.
    """

    decision: AccessDecision
    evidence_withheld_at_rank: int | None = None
    should_refuse: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "should_refuse": self.should_refuse,
            "reason": self.reason,
            "evidence_withheld_at_rank": self.evidence_withheld_at_rank,
            **self.decision.to_dict(),
        }


def authorise(
    candidates: Sequence,
    principal: Principal,
    *,
    tenant_key: str = "tenant",
    protected_rank: int = 1,
) -> AuthorisationOutcome:
    """Authorise, and decide whether the answer must be refused.

    `protected_rank` is how deep the protected zone reaches. A candidate within
    the first `protected_rank` positions being withheld means the retriever's own
    judgement was that this was the answer, and substituting a lower-ranked
    document is a factual substitution -- so refuse. A withheld candidate below
    that depth was not the answer, and saying nothing about it costs nothing.

    Defaults to 1, i.e. only the top hit is protected. Tightening it to the whole
    `final_k` window trades answer rate for safety and is the right knob for a
    deployment where the documents are uniformly sensitive; the parameter exists
    so that trade-off is explicit rather than buried in a threshold.
    """
    decision = filter_candidates(candidates, principal, tenant_key=tenant_key)

    withheld_at_rank: int | None = None
    for index, candidate in enumerate(candidates, start=1):
        if candidate in decision.withheld:
            withheld_at_rank = index
            break

    if withheld_at_rank is not None and withheld_at_rank <= protected_rank:
        return AuthorisationOutcome(
            decision=decision,
            evidence_withheld_at_rank=withheld_at_rank,
            should_refuse=True,
            reason=(
                f"the highest-ranked evidence for this query is restricted to the "
                f"caller's tenant(s); answering from a lower-ranked document would "
                f"substitute a weaker source for the real one"
            ),
        )

    if decision.withheld_anything:
        return AuthorisationOutcome(
            decision=decision,
            evidence_withheld_at_rank=withheld_at_rank,
            should_refuse=False,
            reason=(
                f"{decision.withheld_count} lower-ranked candidate(s) were withheld "
                f"as outside the caller's tenant scope; readable evidence remains "
                f"at the top of the ranking"
            ),
        )

    return AuthorisationOutcome(
        decision=decision,
        evidence_withheld_at_rank=None,
        should_refuse=False,
        reason="all candidates are within the caller's scope",
    )
