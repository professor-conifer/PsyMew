"""Structured MCTS search data for LLM prompt injection.

Extracts visit counts, average scores, and determinization probabilities
from PokeEngine MCTS results and formats them as human-readable strategic
data that the LLM can incorporate into its distribution-based reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum number of top options to show in the prompt block
_MAX_OPTIONS_SHOWN = 8


@dataclass
class MctsMoveStats:
    """Per-option MCTS statistics from a single determinization."""

    move_choice: str  # "move earthquake" or "switch garchomp"
    visits: int
    total_score: float
    visit_pct: float  # visits / total_visits in this determinization

    @property
    def avg_score(self) -> float:
        """Average outcome score for this option (0=loss, 1=win)."""
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits


@dataclass
class DeterminizationResult:
    """MCTS results for one opponent-team determinization."""

    sample_chance: float  # probability of this opponent-set configuration
    side_one_stats: list[MctsMoveStats] = field(default_factory=list)
    total_visits: int = 0


@dataclass
class MctsSearchData:
    """Aggregated MCTS results across all determinizations.

    Provides both a formatted prompt block for the LLM and a blended
    probability distribution usable as a fallback if the LLM is unavailable.
    """

    determinizations: list[DeterminizationResult] = field(default_factory=list)
    blended_policy: dict[str, float] = field(default_factory=dict)
    search_time_ms: int = 0
    num_determinizations: int = 0
    total_rollouts: int = 0

    def to_prompt_block(self, max_options: int = _MAX_OPTIONS_SHOWN) -> str:
        """Format MCTS data as a strategic insights section for the LLM prompt.

        Produces a compact but information-rich block showing per-option
        visit percentages and average scores, plus context about the
        search methodology.
        """
        if not self.blended_policy:
            return ""

        lines = []
        lines.append(
            f"MCTS SEARCH INSIGHTS ({self.num_determinizations} determinizations, "
            f"{self.search_time_ms}ms/search, {self.total_rollouts:,} total rollouts):"
        )

        # Sort blended policy by weight descending, show top options
        sorted_policy = sorted(
            self.blended_policy.items(), key=lambda x: x[1], reverse=True
        )

        for i, (choice, weight) in enumerate(sorted_policy[:max_options]):
            pct = weight * 100
            # Look up average score from individual determinizations
            avg_scores = []
            for det in self.determinizations:
                for stat in det.side_one_stats:
                    if stat.move_choice == choice and stat.visits > 0:
                        avg_scores.append(stat.avg_score)
            avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
            lines.append(f"  {choice}: {pct:.1f}% visits, avg score {avg:.3f}")

        if len(sorted_policy) > max_options:
            remaining = len(sorted_policy) - max_options
            lines.append(f"  ... and {remaining} more options with lower visit counts")

        lines.append("")
        lines.append(
            "These results come from game-tree search across possible opponent "
            "team configurations. Visit % = how often the search engine chose "
            "this option. Avg score = average outcome (0=loss, 1=win) when it "
            "was played."
        )
        lines.append("")
        lines.append(
            "NOTE: The search engine makes assumptions about unrevealed sets, "
            "abilities, and items. Its visit distribution may overfit to common "
            "configurations. Use this data as strategic input alongside your own "
            "reasoning about what the opponent is actually likely to have."
        )

        return "\n".join(lines)

    def get_blended_distribution(
        self, legal_options: list[str] | None = None
    ) -> dict[str, float]:
        """Return normalized probability distribution for sampling.

        Filters to only the given legal_options if provided, then re-normalizes.
        """
        if not self.blended_policy:
            return {}

        if legal_options is not None:
            filtered = {
                k: v
                for k, v in self.blended_policy.items()
                if k in legal_options
            }
        else:
            filtered = dict(self.blended_policy)

        total = sum(filtered.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in filtered.items()}


def build_mcts_search_data(
    mcts_results: list[tuple[object, float, int]],
    search_time_ms: int,
) -> MctsSearchData:
    """Build MctsSearchData from raw MCTS results.

    Parameters
    ----------
    mcts_results : list of (MctsResult, sample_chance, index)
        Raw results from parallel MCTS searches.
    search_time_ms : int
        Search time per determinization in milliseconds.

    Returns
    -------
    MctsSearchData
        Structured data ready for prompt formatting and distribution blending.
    """
    determinizations = []
    blended: dict[str, float] = {}
    total_rollouts = 0

    for mcts_result, sample_chance, _index in mcts_results:
        total_visits = mcts_result.total_visits
        total_rollouts += total_visits

        side_stats = []
        for s1_option in mcts_result.side_one:
            if total_visits > 0:
                visit_pct = s1_option.visits / total_visits
            else:
                visit_pct = 0.0

            stats = MctsMoveStats(
                move_choice=s1_option.move_choice,
                visits=s1_option.visits,
                total_score=s1_option.total_score,
                visit_pct=visit_pct,
            )
            side_stats.append(stats)

            # Accumulate into blended policy (visit-weighted by sample chance)
            blended[s1_option.move_choice] = blended.get(
                s1_option.move_choice, 0.0
            ) + (sample_chance * visit_pct)

        det = DeterminizationResult(
            sample_chance=sample_chance,
            side_one_stats=side_stats,
            total_visits=total_visits,
        )
        determinizations.append(det)

    # Normalize blended policy
    total_weight = sum(blended.values())
    if total_weight > 0:
        blended = {k: v / total_weight for k, v in blended.items()}

    return MctsSearchData(
        determinizations=determinizations,
        blended_policy=blended,
        search_time_ms=search_time_ms,
        num_determinizations=len(determinizations),
        total_rollouts=total_rollouts,
    )
