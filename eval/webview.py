"""
Present Barplots from graphs.py and Table from inspect_eval.py on a HTML page for even better overview
"""

import argparse
from pathlib import Path

from flask import Flask
from flask import render_template_string
from prettytable import PrettyTable

from eval.graphs import generate_barplots_for_test_series
from eval.inspect_eval import IDENTIFIER_TO_CPV
from eval.inspect_eval import evaluate_test_series
from eval.inspect_eval import generate_results_table
from eval.inspect_eval import merge_pretty_tables_to_pandas

app = Flask(__name__, static_folder="plots")

# --- Placeholder vars you can update ---
SUMMARY_TITLE = "Summary of Analyzer and Verifier Results"
SUMMARY_DESC = "This barplot shows the aggregated results across the whole set of of given test series."

COMMIT_TITLE = "Commit-Specific Results"
COMMIT_DESC = "Each barplot below corresponds to a specific commit."

# paths to plot images (assume you saved them already)
SUMMARY_PLOT = "barplot-summary.png"

# store table HTML globally (set in main)
TABLES_HTML: list[str] = []


TEMPLATE_STRING_NGINX = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Barplots Viewer</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 40px;
    }

    h2 { margin-top: 40px; }

    .plot {
      margin: 20px 0;
      width: 100%;
    }

    .plot img {
      width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
    }

    /* Pretty table styling */
    .pretty-table {
      margin: 20px auto;
      border-collapse: collapse;
      width: 90%;
      text-align: center;
      font-size: 0.95rem;
    }

    .pretty-table th, .pretty-table td {
      border: 1px solid #999;
      padding: 8px 12px;
    }

    .pretty-table th {
      background-color: #eee;
      font-weight: bold;
    }

    /* Striped rows for readability */
    .pretty-table tr:nth-child(even) {
      background-color: #f9f9f9;
    }

    /* Hover effect */
    .pretty-table tr:hover {
      background-color: #d3ebf9;
    }

    /* Responsive legend or figure containers */
    figure {
      width: 100%;
      margin: 0 auto;
    }

  </style>
</head>
<body>
  <h1>{{ summary_title }}</h1>
  <p>{{ summary_desc }}</p>

  <div class="plot">
    <h3>{{ summary_plot }}</h3>
    <img src="{{ url_for('static', filename=summary_plot) }}">
  </div>

  <h2>Results Tables</h2>
  {{ table_html|safe }}

  <h2>{{ commit_title }}</h2>
  <p>{{ commit_desc }}</p>

  {% for name, img in commit_plots %}
    <div class="plot">
      <h3>{{ name }}</h3>
      <img src="{{ url_for('static', filename=img) }}">
    </div>
  {% endfor %}
</body>
</html>
"""

TEMPLATE_STRING_NON_NGINX = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Barplots Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; }
    h2 { margin-top: 40px; }
    .plot { margin: 20px 0; }

    /* Pretty table styling */
    .pretty-table {
      margin: 20px auto;
      border-collapse: collapse;
      width: 90%;
      text-align: center;
    }

    .pretty-table th, .pretty-table td {
      border: 1px solid #999;
      padding: 8px 12px;
    }

    .pretty-table th {
      background-color: #eee;
      font-weight: bold;
    }

    /* Striped rows for readability */
    .pretty-table tr:nth-child(even) {
      background-color: #f9f9f9;
    }

    /* Hover effect */
    .pretty-table tr:hover {
      background-color: #d3ebf9;
    }
  </style>
</head>

<body>
  <h1>{{ summary_title }}</h1>
  <p>{{ summary_desc }}</p>
  <div class="plot">
    <h3>{{ summary_plot }}</h3>
    <img src="{{ url_for('static', filename=summary_plot) }}" width="800">
  </div>

  <h2>{{ commit_title }}</h2>
  <p>{{ commit_desc }}</p>

  {% for name, img in commit_plots %}
    <div class="plot">
        <h3>{{ name }}</h3>
        <img src="{{ url_for('static', filename=img) }}">
    </div>
  {% endfor %}
</body>
</html>
        """


def main(
    test_series_dirs: list[Path],
    image_dir: Path,
    port: int,
    is_nginx: bool = True,
    plots_already_built: bool = False,
) -> None:
    """
    Prepare Pretty Table and Barplots before launching website
    """
    # generate pretty tables
    if is_nginx:
        template_str = TEMPLATE_STRING_NGINX
        tables_html: list[str] = []
        pretty_tables: list[PrettyTable] = []
        test_series_names: list[str] = []
        for test_series in test_series_dirs:
            data = evaluate_test_series(test_series, False)

            test_series_names.append(test_series.name)
            table = generate_results_table(data)
            pretty_tables.append(table)
            tables_html.append(table.get_html_string(attributes={"class": "pretty-table"}))

        multi_index_table = merge_pretty_tables_to_pandas(pretty_tables, test_series_names)
        tables_html_str = multi_index_table.to_html(classes="pretty-table", border=0)
    else:
        template_str = TEMPLATE_STRING_NON_NGINX
        tables_html_str = ""

    # generate barplots into image_dir
    if not plots_already_built:
        generate_barplots_for_test_series(image_dir, test_series_dirs, True)

    # Update Flask static folder to point to image_dir
    app.static_folder = str(image_dir)

    # Prepare commit plots (exclude summary)
    commit_plots_files = [f.name for f in image_dir.glob("*.png") if f.name != SUMMARY_PLOT]
    name_and_plot: dict[str, str] = {}
    if is_nginx:  # Pretty 'CPV xx: commit' titles for nginx
        for commit, cpv in IDENTIFIER_TO_CPV.items():
            cur_name = f"CPV {cpv}: {commit}"
            cur_plot = next((fname for fname in commit_plots_files if commit in fname), None)
            if cur_plot:
                name_and_plot[cur_name] = cur_plot
    else:  # generic file name titles for other types of test series
        for file in commit_plots_files:
            name_and_plot[file] = file

    @app.route("/")
    def index() -> str:
        return render_template_string(
            template_str,
            summary_title=SUMMARY_TITLE,
            summary_desc=SUMMARY_DESC,
            summary_plot=SUMMARY_PLOT,
            table_html=tables_html_str,
            commit_title=COMMIT_TITLE,
            commit_desc=COMMIT_DESC,
            commit_plots=sorted(name_and_plot.items(), key=lambda x: x[0]),  # sort by name
        )

    app.run(debug=True, port=port, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Locally hosted website for pretty evaluation visualization. *happy Mischa noise*"
    )
    parser.add_argument("output_path", type=Path, help="Path to save barplots to")
    parser.add_argument("input_dirs", type=Path, nargs="+", help="Path to test series.")
    parser.add_argument("port", type=int, help="Port to server the website")
    parser.add_argument(
        "--non-nginx",
        action="store_true",
        help="At least one of the provided test series is not based in the AIxCC Nginx CP",
    )
    parser.add_argument(
        "--plots-already-built",
        action="store_true",
        help="Barplots to be displayed are already build and inside the output_path",
    )

    args = parser.parse_args()
    output_dir = args.output_path.resolve()

    main(
        test_series_dirs=args.input_dirs,
        image_dir=output_dir,
        port=args.port,
        is_nginx=(not args.non_nginx),
        plots_already_built=args.plots_already_built,
    )
