"""Optional evaluation CLI adapters for the minimal Kernos install."""

from __future__ import annotations


def _missing_examples_message(command: str, error: ModuleNotFoundError) -> SystemExit:
    """Build an actionable error for an unavailable evaluation dependency."""
    return SystemExit(
        f"{command} requires the optional evaluation dependencies; "
        f"install them with `pip install 'kernos[examples]'` ({error.name} is missing)."
    )


def eval_main() -> None:
    """Run the optional ``kernos-eval`` command with a clear install error."""
    try:
        from examples.evaluation.cli import main

        main()
    except ModuleNotFoundError as error:
        raise _missing_examples_message("kernos-eval", error) from error


def analyze_main() -> None:
    """Run the optional ``kernos-analyze`` command with a clear install error."""
    try:
        from examples.evaluation.analyze_cli import main

        main()
    except ModuleNotFoundError as error:
        raise _missing_examples_message("kernos-analyze", error) from error
