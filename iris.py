"""
IRIS — Identity-Regulated Inference System

Computes the regulatory signal T from incoming evidence and the CORE belief
graph, then returns a decision (assimilate / accommodate / reject) and
updates the assimilative-inertia parameter α via the identity-buffering
mechanism.

T1 is SIGNED: T1 = confidence_current(bi) - confidence_posterior(bi | e).
Positive T1 means the evidence DECREASED confidence (disconfirming /
threatening); negative T1 means the evidence INCREASED confidence
(confirming). This matters because only disconfirming pressure should be
able to push the system toward accommodation or rejection — confirmatory
evidence, however large its raw impact on model logits, poses no identity-
integrity threat and should pull the system toward assimilation instead.

"Rejection" replaces what was previously called "branching." Rejection is
NOT structural isolation (no adapter forking is implemented in this study)
- it is refusal: when integrity threat is severe enough (high T1 combined
with high centrality x cascade ratio), the system does not update at all.
No fine-tuning runs, no belief confidence changes, the day's evidence is
processed in conversation but leaves no lasting trace on CORE or on model
weights. This is grounded in defensive-avoidance / motivated-reasoning
phenomena (Sherman & Cohen, 2006), where threatening information is not
merely deprioritized but not integrated at all.

Assimilation, in this version, is NOT a null operation. When evidence is
confirmatory (negative T1) and/or causes negligible disruption, IRIS
applies a small, bounded confidence INCREASE scaled by the magnitude of
T1, reflecting that successful assimilation reinforces the existing
schema rather than leaving it untouched - consistent with general schema-
theoretic findings that repeated assimilation strengthens belief.

α is owned exclusively by IRISController and is the only place it is set or
modified in the entire codebase.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from core.core import CORE


LOG_PATH = os.path.join(os.path.dirname(__file__), "iris_log.json")

# Candidate surface forms for "yes" — we take the union of their token ids.
_YES_VARIANTS = ("Yes", "yes", "YES")


class IRISController:
    """
    Stateful IRIS controller.  One instance per experiment session; α
    persists across ``run`` calls within that session.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        tau_a: float = 0.0,
        tau_b: float = 0.2,
        beta: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        alpha : float
            Assimilative inertia, initialised at 0.7.
        tau_a : float
            Lower decision threshold (assimilation boundary), fixed at 0.0.
        tau_b : float
            Upper decision threshold (accommodation/rejection boundary),
            fixed at 0.2.
        beta : float
            Identity-buffering scaling factor, fixed at 0.1.
        gamma : float
            Assimilation confidence-reinforcement scaling factor. On an
            assimilate decision, confidence increases by gamma * |T1|,
            where |T1| reflects how much the (confirmatory or negligibly
            disruptive) evidence actually engaged with the belief.
        """
        self.alpha = alpha
        self.tau_a = tau_a
        self.tau_b = tau_b
        self.beta = beta
        self.gamma = gamma

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------

    def get_confidence(self, model, tokenizer, belief_statement, context=""):
        """
        Extract the model's confidence that *belief_statement* is true.

        A binary prompt is constructed; the softmax probability of the
        "Yes" token (handling capitalisation variants) is returned.

        Parameters
        ----------
        model : PreTrainedModel
            The loaded causal-LM (Gemma E4B or compatible).
        tokenizer : PreTrainedTokenizer
            Matching tokenizer with ``padding_side="left"`` recommended.
        belief_statement : str
            Natural-language proposition to evaluate.
        context : str, optional
            Evidence text prepended to the prompt.  Empty string means
            the prompt uses only the belief statement (no evidence).

        Returns
        -------
        float
            Softmax probability of "Yes", in [0, 1].  Returns 0.0 if
            the "Yes" token is absent from the vocabulary.
        """
        if context:
            user_content = (
                f"Context: {context}\n\n"
                f"Given this context, is the following statement true about you? "
                f"Answer only Yes or No.\n\nStatement: {belief_statement}"
            )
        else:
            user_content = (
                f"Is the following statement true about you? "
                f"Answer only Yes or No.\n\nStatement: {belief_statement}"
            )

        messages = [{"role": "user", "content": user_content}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[0, -1, :]

        yes_ids = []
        for variant in _YES_VARIANTS:
            ids = tokenizer.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                yes_ids.append(ids[0])

        if not yes_ids:
            return 0.0

        probs = torch.softmax(next_token_logits, dim=-1)
        yes_prob = sum(probs[i].item() for i in set(yes_ids))
        return float(min(yes_prob, 1.0))

    # ------------------------------------------------------------------
    # T1 — evidential disruption
    # ------------------------------------------------------------------

    def compute_t1(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        belief_statement: str,
        evidence: str,
    ) -> float:
        """
        Compute T1 — the signed plasticity demand for a single belief.

        T1 = P(belief is true | no evidence) − P(belief is true | evidence)

        T1 is SIGNED, not an absolute value. Positive T1 means the evidence
        DECREASED the model's confidence in the belief (disconfirming /
        threatening evidence — genuine plasticity demand). Negative T1
        means the evidence INCREASED confidence (confirming evidence —
        no identity-integrity threat, even if the magnitude of logit
        shift is large). This distinction is what allows T to correctly
        pull toward assimilation under confirmation and toward
        accommodation/rejection only under genuine disconfirmation.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizer
        belief_statement : str
            Natural-language proposition (node label).
        evidence : str
            New information to evaluate against the belief.

        Returns
        -------
        float
            Signed change in model confidence, in [-1, 1]. Positive =
            disconfirming, negative = confirming.
        """
        confidence_current = self.get_confidence(model, tokenizer, belief_statement)
        confidence_posterior = self.get_confidence(
            model, tokenizer, belief_statement, context=evidence
        )
        return confidence_current - confidence_posterior

    # ------------------------------------------------------------------
    # IRIS signal
    # ------------------------------------------------------------------

    def compute_iris(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        core: "CORE",
        belief_id: str,
        evidence: str,
    ) -> dict:
        """
        Compute the full IRIS regulatory signal T.

        T = (1 − α) · T1 − α · (T2 × T3)

        - T1 = signed plasticity demand (computed via ``compute_t1``).
          Positive = disconfirming evidence, negative = confirming evidence.
        - T2 = identity centrality of the belief (``core.get_centrality``)
        - T3 = cascade ratio of the belief (``core.cascade_ratio``)
        - α  = ``self.alpha`` at call time

        Bounds: with T1 ∈ [-1, 1] and T2, T3 ∈ [0, 1], T ∈ [-1, 1-α].
        Confirming evidence (negative T1) always pulls T downward toward
        assimilation and can never alone push T into the rejection zone,
        regardless of T2 or T3, since both terms in T can only push T
        further negative when T1 is negative.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizer
        core : CORE
            Populated CORE belief graph.
        belief_id : str
            ID of the belief node to evaluate.
        evidence : str
            Incoming evidence string.

        Returns
        -------
        dict
            Keys: T, T1, T2, T3, alpha.

        Raises
        ------
        ValueError
            If *belief_id* is not found in the CORE graph.
        """
        node = core.get_node(belief_id)  # raises ValueError if not found
        belief_statement = node.label

        t1 = self.compute_t1(model, tokenizer, belief_statement, evidence)
        t2 = core.get_centrality(belief_id)
        t3 = core.cascade_ratio(belief_id)

        T = (1.0 - self.alpha) * t1 - self.alpha * (t2 * t3)

        return {
            "T": T,
            "T1": t1,
            "T2": t2,
            "T3": t3,
            "alpha": self.alpha,
        }

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def iris_decision(self, T: float) -> str:
        """
        Map the regulatory signal T to a decision string.

        - T < τ_A  → "assimilate"
        - τ_A ≤ T < τ_B → "accommodate"
        - T ≥ τ_B  → "reject"

        Because T1 is signed, this boundary structure now has a clean
        interpretation: confirming evidence (negative T1) reliably lands
        in "assimilate". Disconfirming evidence (positive T1) lands in
        "accommodate" if the structural cost (T2 x T3) is low enough to
        make restructuring safe, or "reject" if the belief is too
        centrally important / too cascading to safely restructure given
        how much the evidence disconfirmed it. "reject" is reachable only
        through genuine disconfirmation, never through confirmation,
        regardless of T2 or T3.

        Parameters
        ----------
        T : float
            IRIS regulatory signal.

        Returns
        -------
        str
            One of "assimilate", "accommodate", "reject".
        """
        if T < self.tau_a:
            return "assimilate"
        if T < self.tau_b:
            return "accommodate"
        return "reject"

    # ------------------------------------------------------------------
    # Identity-buffering α update
    # ------------------------------------------------------------------

    def update_alpha(self, T: float, decision: str) -> float:
        """
        Update ``self.alpha`` via the identity-buffering mechanism.

        δ = β · |T|

        - Assimilation:        α ← clamp(α − δ, 0, 1)
        - Accommodation / reject: α ← clamp(α + δ, 0, 1)

        Parameters
        ----------
        T : float
            IRIS regulatory signal from the current call.
        decision : str
            Decision string returned by ``iris_decision``.

        Returns
        -------
        float
            Updated ``self.alpha``.
        """
        delta = self.beta * abs(T)
        if decision == "assimilate":
            self.alpha = max(0.0, min(1.0, self.alpha - delta))
        else:
            self.alpha = max(0.0, min(1.0, self.alpha + delta))
        return self.alpha

    # ------------------------------------------------------------------
    # Assimilation confidence reinforcement
    # ------------------------------------------------------------------

    def compute_assimilation_confidence_bump(self, t1: float) -> float:
        """
        Compute the confidence increase to apply on an "assimilate"
        decision.

        bump = γ · |T1|

        Assimilation is not a null operation: when evidence is
        successfully assimilated (confirming, or negligibly disruptive),
        the existing belief is reinforced rather than left untouched,
        consistent with schema-theoretic findings that repeated
        assimilation strengthens belief. |T1| is used (rather than signed
        T1) because on an assimilate decision T1 will typically be
        negative or near zero, and the magnitude of that value reflects
        how much the evidence actually engaged with the belief at all,
        regardless of which direction it nudged it.

        Parameters
        ----------
        t1 : float
            The signed T1 value from the same IRIS call that produced
            the "assimilate" decision.

        Returns
        -------
        float
            The confidence increase to apply, non-negative.
        """
        return self.gamma * abs(t1)

    # ------------------------------------------------------------------
    # Top-level runner
    # ------------------------------------------------------------------

    def run(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
        core: "CORE",
        belief_id: str,
        evidence: str,
        day: int | None = None,
    ) -> dict:
        """
        Execute one full IRIS cycle and append the result to iris_log.json.

        Steps:
        1. Compute the IRIS regulatory signal T.
        2. Derive the decision from T.
        3. Record α before updating.
        4. Update α via identity-buffering.
        5. If decision is "assimilate", compute the confidence
           reinforcement bump (caller is responsible for applying it to
           CORE — this method does not mutate CORE directly).
        6. Persist the log entry.

        Parameters
        ----------
        model : PreTrainedModel
        tokenizer : PreTrainedTokenizer
        core : CORE
            Populated CORE belief graph.
        belief_id : str
            ID of the belief node to evaluate.
        evidence : str
            Incoming evidence string.
        day : int, optional
            Day number in the overall experiment this call corresponds to.
            Logged alongside the rest of the entry so daily IRIS decisions
            can be queried directly without reconstructing them from
            timestamps. None if not applicable (e.g. ad hoc/test calls).

        Returns
        -------
        dict
            Keys: belief_id, evidence, day, T, T1, T2, T3,
                  alpha_before, alpha_after, decision,
                  assimilation_confidence_bump (0.0 if decision is not
                  "assimilate"), timestamp.

        Raises
        ------
        ValueError
            If *belief_id* is not found in the CORE graph.
        """
        iris_signal = self.compute_iris(model, tokenizer, core, belief_id, evidence)
        T = iris_signal["T"]
        t1 = iris_signal["T1"]
        decision = self.iris_decision(T)

        alpha_before = self.alpha
        alpha_after = self.update_alpha(T, decision)

        assimilation_confidence_bump = (
            self.compute_assimilation_confidence_bump(t1)
            if decision == "assimilate"
            else 0.0
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        entry = {
            "belief_id": belief_id,
            "evidence": evidence,
            "day": day,
            "T": iris_signal["T"],
            "T1": iris_signal["T1"],
            "T2": iris_signal["T2"],
            "T3": iris_signal["T3"],
            "alpha_before": alpha_before,
            "alpha_after": alpha_after,
            "decision": decision,
            "assimilation_confidence_bump": assimilation_confidence_bump,
            "timestamp": timestamp,
        }

        _append_log(entry)
        return entry


# ------------------------------------------------------------------
# Log helper
# ------------------------------------------------------------------

def _append_log(entry: dict) -> None:
    """Append *entry* to iris_log.json, creating the file if needed."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            try:
                log = json.load(fh)
            except json.JSONDecodeError:
                log = []
    else:
        log = []

    log.append(entry)

    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)