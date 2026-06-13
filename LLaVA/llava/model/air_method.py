from __future__ import annotations

# Public names for reports, tables, and result directory suffixes
AIR_METHOD_NAME: str = "AIR"
AIR_METHOD_FULL_NAME: str = "AIR (Attention-Invariance Regularization)"

# Internal module identifiers (for method sections)
AIR_MODULE_QK_SPECTRAL_RESCALE: str = "qk_spectral_energy_rescale_pre_softmax"
AIR_MODULE_MODALITY_REBALANCE: str = "modality_rebalance_pre_softmax"
AIR_MODULE_CROSS_HEAD_LENS: str = "cross_head_lens_post_mask_pre_softmax"
AIR_MODULE_CONDITIONAL_ADHH: str = "conditional_adhh_post_softmax"
AIR_MODULE_VARIANCE_PROJECTION: str = "variance_constrained_projection_post_softmax"

# Dynamic vision boost schedules for modality rebalancing (see module doc above)
AIR_GAMMA_SCHEDULES: tuple = ("const", "exp", "log")


def air_method_one_liner() -> str:
    """One-line summary for logs or figure captions."""
    return (
        "AIR: mid-layer modality rebalancing, cross-head vision lens, conditional AD-HH, and "
        "variance-constrained projection regularization (paper §5.2) to reduce language-prior "
        "dominance, hallucination-head text over-reliance, and attention over-concentration."
    )


def air_problems_addressed() -> tuple[str, ...]:
    """Phenomena AIR targets (for motivation paragraphs)."""
    return (
        "Modality imbalance: weak image-token attention vs. system text at decode steps.",
        "Cross-head inconsistency: heads use visual keys in ways that hurt image alignment.",
        "Hallucination heads over-aggregate post-image text, amplifying non-visual content.",
        "Over-concentrated or diffuse attention; unstable decoding distributions.",
    )
