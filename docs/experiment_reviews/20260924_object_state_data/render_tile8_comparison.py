"""Render one-reference-action-per-inference guide8 comparison."""

from pathlib import Path

from render_repeat8_comparison import main


if __name__ == '__main__':
    main(out=Path(__file__).resolve().parent /
         'corrected_direct_vs_reference/tile8_advance_per_call', source_repeats=2)
