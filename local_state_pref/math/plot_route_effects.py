from __future__ import annotations

# Example: plot current MATH BCE-KL together with the old SFT/AS-KL baseline.
# DPO is intentionally not included.
#
#   python plot_route_effects.py \
#     --metadata BCE-KL=/path/to/bce_kl_metadata.csv \
#     --metadata SFT=/path/to/sft_metadata.csv \
#     --expert_csv /path/to/expert_results.csv \
#     --oracle_source SFT \
#     --out_dir /path/to/route_curve_with_sft

# Outputs:
#   route_effect_summary.csv
#   route_effect_summary.txt
#   route_distribution.png
#   route_outcomes.png
#   route_overall_metrics.png
#   budget_curve.png
import importlib.util
from pathlib import Path


_CODE_PLOT_PATH = Path(__file__).resolve().parents[1] / "code" / "plot_route_effects.py"
_SPEC = importlib.util.spec_from_file_location("_code_plot_route_effects", _CODE_PLOT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load shared route plotting implementation: {_CODE_PLOT_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ROUTE_ORDER = _MODULE.ROUTE_ORDER
ROUTE_DISPLAY = _MODULE.ROUTE_DISPLAY
ROUTE_COLORS = _MODULE.ROUTE_COLORS

parse_metadata_spec = _MODULE.parse_metadata_spec
robust_read_csv = _MODULE.robust_read_csv
normalize_route = _MODULE.normalize_route
to_float = _MODULE.to_float
load_expert_map = _MODULE.load_expert_map
load_metadata_rows = _MODULE.load_metadata_rows
is_deferred_row = _MODULE.is_deferred_row
compute_system_success_rate = _MODULE.compute_system_success_rate
compute_budget_curve = _MODULE.compute_budget_curve
select_oracle_comparison = _MODULE.select_oracle_comparison
compute_plot_ylim = _MODULE.compute_plot_ylim
compute_route_summary = _MODULE.compute_route_summary
write_summary_csv = _MODULE.write_summary_csv
write_summary_report = _MODULE.write_summary_report
plot_route_distribution = _MODULE.plot_route_distribution
plot_route_outcomes = _MODULE.plot_route_outcomes
plot_overall_metrics = _MODULE.plot_overall_metrics
plot_budget_curves = _MODULE.plot_budget_curves
main = _MODULE.main


if __name__ == "__main__":
    main()
