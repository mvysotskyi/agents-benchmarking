"""Token usage tracking for the Claude Computer Use Agent."""

from dataclasses import dataclass

from computer_use.config import ModelSpec


@dataclass
class IterationUsage:
    """Token usage and timing for a single iteration."""

    iteration: int
    input_tokens: int
    output_tokens: int
    step_elapsed: float = 0.0
    agent_elapsed: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens for this iteration."""
        return self.input_tokens + self.output_tokens


class TokenUsageTracker:
    """Tracks token usage across agent iterations."""

    def __init__(self, model_spec: ModelSpec) -> None:
        """Initialize tracker with model pricing info.

        Args:
            model_spec: Model specification with pricing and context limits.
        """
        self.model_spec = model_spec
        self.iterations: list[IterationUsage] = []
        self._peak_context: int = 0

    def record_iteration(
        self,
        iteration: int,
        usage: dict[str, int],
        step_elapsed: float = 0.0,
        agent_elapsed: float = 0.0,
    ) -> IterationUsage:
        """Record token usage and timing for an iteration.

        Args:
            iteration: Iteration number.
            usage: Usage dict from API response with input_tokens and output_tokens.
            step_elapsed: Seconds spent on the LLM call for this iteration.
            agent_elapsed: Total seconds for this iteration (LLM + action exec).

        Returns:
            IterationUsage for this iteration.
        """
        iter_usage = IterationUsage(
            iteration=iteration,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            step_elapsed=step_elapsed,
            agent_elapsed=agent_elapsed,
        )
        self.iterations.append(iter_usage)

        if iter_usage.input_tokens > self._peak_context:
            self._peak_context = iter_usage.input_tokens

        return iter_usage

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all iterations."""
        return sum(i.input_tokens for i in self.iterations)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all iterations."""
        return sum(i.output_tokens for i in self.iterations)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def peak_context(self) -> int:
        """Peak context size observed."""
        return self._peak_context

    def calculate_cost(self) -> float:
        """Calculate total cost in USD.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (self.total_input_tokens / 1_000_000) * self.model_spec.input_price_per_mtok
        output_cost = (self.total_output_tokens / 1_000_000) * self.model_spec.output_price_per_mtok
        return input_cost + output_cost

    def calculate_iteration_cost(self, iter_usage: IterationUsage) -> float:
        """Calculate cost for a single iteration.

        Args:
            iter_usage: Iteration usage data.

        Returns:
            Estimated cost in USD for this iteration.
        """
        input_cost = (iter_usage.input_tokens / 1_000_000) * self.model_spec.input_price_per_mtok
        output_cost = (iter_usage.output_tokens / 1_000_000) * self.model_spec.output_price_per_mtok
        return input_cost + output_cost

    def get_current_context_size(self) -> int:
        """Get the most recent input token count as context size estimate.

        Returns:
            Current context size in tokens.
        """
        if not self.iterations:
            return 0
        return self.iterations[-1].input_tokens

    def get_context_fill_percentage(self) -> float:
        """Calculate percentage of context window filled.

        Returns:
            Context fill percentage (0-100).
        """
        current = self.get_current_context_size()
        return (current / self.model_spec.context_window) * 100

    def get_peak_context_percentage(self) -> float:
        """Calculate peak context percentage observed.

        Returns:
            Peak context fill percentage (0-100).
        """
        return (self._peak_context / self.model_spec.context_window) * 100

    def get_summary_stats(self) -> dict[str, float | int]:
        """Build a flat dict of cumulative/max token and timing stats.

        Returns:
            Dict with keys matching the gpt54 ``stats.*`` schema:
            ``cum_steps``, ``cum_input_tokens``, ``max_input_tokens``,
            ``cum_output_tokens``, ``max_output_tokens``, ``cum_total_tokens``,
            ``max_total_tokens``, ``cum_step_elapsed``, ``max_step_elapsed``,
            ``cum_agent_elapsed``, ``max_agent_elapsed``.
        """
        if not self.iterations:
            return {
                "cum_steps": 0,
                "cum_input_tokens": 0,
                "max_input_tokens": 0,
                "cum_output_tokens": 0,
                "max_output_tokens": 0,
                "cum_total_tokens": 0,
                "max_total_tokens": 0,
                "cum_step_elapsed": 0.0,
                "max_step_elapsed": 0.0,
                "cum_agent_elapsed": 0.0,
                "max_agent_elapsed": 0.0,
            }

        return {
            "cum_steps": len(self.iterations),
            "cum_input_tokens": self.total_input_tokens,
            "max_input_tokens": max(i.input_tokens for i in self.iterations),
            "cum_output_tokens": self.total_output_tokens,
            "max_output_tokens": max(i.output_tokens for i in self.iterations),
            "cum_total_tokens": self.total_tokens,
            "max_total_tokens": max(i.total_tokens for i in self.iterations),
            "cum_step_elapsed": sum(i.step_elapsed for i in self.iterations),
            "max_step_elapsed": max(i.step_elapsed for i in self.iterations),
            "cum_agent_elapsed": sum(i.agent_elapsed for i in self.iterations),
            "max_agent_elapsed": max(i.agent_elapsed for i in self.iterations),
        }

    def format_context_window(self) -> str:
        """Format context window size for display.

        Returns:
            Human-readable context window size (e.g., '200K').
        """
        window = self.model_spec.context_window
        if window >= 1_000_000:
            return f"{window // 1_000_000}M"
        if window >= 1_000:
            return f"{window // 1_000}K"
        return str(window)
