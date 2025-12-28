"""tex table printer module"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LATEX_REPLACEMENTS = [
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]


def latex_escape(value: Any) -> str:
    """Escapes string for latex"""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    for old, new in LATEX_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def format_latex_row(values: list[Any]) -> str:
    """method to format latex row"""
    return " & ".join(str(v) for v in values) + " \\\\"


def get_color_cell(value: float) -> str:
    """maps value to color"""
    if not isinstance(value, (int, float)) or pd.isna(value):
        return latex_escape(value)

    color = ""
    if value == 0:
        color = "myred"
    elif 1 <= value <= 5:
        color = "mylightred"
    elif 6 <= value <= 10:
        color = "myorange"
    elif 11 <= value <= 15:
        color = "myyellow"
    elif 16 <= value <= 20:
        color = "mygreen"

    if color:
        return f"\\cellcolor{{{color}}}{latex_escape(value)}"
    return latex_escape(value)


def generate_csvs(path_to_yaml: str, storage_prefix: str) -> tuple[list[str], list[str]]:
    """
    Parse a YAML file of model evaluations and run inspect_eval for each entry.

    Args:
        path_to_yaml (str): Path to the YAML configuration file.
        storage_prefix (str): Directory prefix where CSVs should be stored.

    Returns:
        tuple[list[str], list[str]]: (list of model names, list of CSV file paths)
    """
    # Ensure the storage prefix directory exists
    storage_dir = Path(storage_prefix)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Load YAML content
    with open(path_to_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_names = []
    csv_paths = []

    for model_name, model_info in data.items():
        data_path = model_info.get("path")
        if not data_path:
            print(f"⚠️ Warning: Missing 'path' for entry '{model_name}', skipping.")
            continue

        csv_path = storage_dir / f"{model_name}.csv"

        # Run the command
        cmd = ["inspect_eval", data_path, "--csv", str(csv_path)]
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Generated CSV for {model_name} → {csv_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate CSV for {model_name}: {e}")
            continue

        model_names.append(model_name)
        csv_paths.append(str(csv_path))

    return model_names, csv_paths


def generate_header(model_names: list[str]) -> str:
    """generates latex table header"""
    latex_top_defs = r"""\tiny
\definecolor{myred}{HTML}{FFC7CE}
\definecolor{mylightred}{HTML}{FFDADA}
\definecolor{myorange}{HTML}{FFEBCC}
\definecolor{myyellow}{HTML}{FFF2CC}
\definecolor{mygreen}{HTML}{D5E8D4}
"""
    num_models = len(model_names)
    cols_after = 2 * num_models  # two columns per model: Verifier and Successful
    col_spec = "|c|c|" + "c|" * cols_after
    header_lines = []
    header_lines.append(latex_top_defs.strip())
    header_lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    header_lines.append("\\hline")

    # first header row: CPV, Function, then a multicolumn for each model
    first_row = "\\multirow{2}{*}{\\textbf{CPV}} & \\multirow{2}{*}{\\textbf{Function}}"
    for m in model_names:
        first_row += f" & \\multicolumn{{2}}{{c|}}{{\\textbf{{{latex_escape(m)}}}}}"
    first_row += " \\\\"
    header_lines.append(first_row)

    last_col = 2 + cols_after
    header_lines.append(f"\\cline{{3-{last_col}}}")

    # second header row: two empty cells (for the multirow CPV and Function), then Verifier/Successful per model
    cols = ["", ""] + (["Verifier", "Successful"] * num_models)
    second_row = " & ".join(cols) + " \\\\"
    header_lines.append(second_row)
    header_lines.append("\\hline")

    return "\n".join(header_lines) + "\n"


def read_csvs(model_names: list[str], csv_files: list[str]) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
    """reads in csv files"""
    aggregated_data: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
    # Loop through files and read them
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        df = df[df["CPV"] != "TOTAL"]  # skip total row if present

        # Ensure correct types
        df[["Total Samples", "Successful Samples", "Vulnerable", "Secure"]] = df[
            ["Total Samples", "Successful Samples", "Vulnerable", "Secure"]
        ].astype(int)

        for _, row in df.iterrows():
            key = (row["CPV"], row["Commit"])
            if key not in aggregated_data:
                aggregated_data[key] = {}
            aggregated_data[key][model_names[i]] = {
                "Secure": row["Secure"],
                "Verifier": row["Vulnerable"],
                "Successful": row["Successful Samples"],
            }
    return aggregated_data


def run(model_names: list[str], csv_files: list[str], standalone: bool) -> None:
    "table generator method"
    latex_rows = []
    # Build the LaTeX table header dynamically based on model names

    latex_header = generate_header(model_names)
    aggregated_data = read_csvs(model_names, csv_files)

    # Sort the keys by CPV
    sorted_keys = sorted(aggregated_data.keys(), key=lambda x: int(x[0]))

    # Initialize totals
    totals = {model: {"Verifier": 0, "Successful": 0} for model in model_names}

    for cpv, func in sorted_keys:
        new_row = [latex_escape(f"{cpv}"), latex_escape(func)]
        for model in model_names:
            data = aggregated_data.get((cpv, func), {}).get(model, {"Verifier": 0, "Successful": 0})
            new_row.extend(
                [
                    get_color_cell(data["Verifier"]),
                    get_color_cell(data["Successful"]),
                ]
            )
            if model in aggregated_data.get((cpv, func), {}):
                totals[model]["Verifier"] += data["Verifier"]
                totals[model]["Successful"] += data["Successful"]
        latex_rows.append(format_latex_row(new_row))

    # Add total row
    total_row = [latex_escape("TOTAL"), "-"]
    for model in model_names:
        t = totals[model]
        total_row.extend([latex_escape(t["Verifier"]), latex_escape(t["Successful"])])
    latex_rows.append(r"\hline")
    latex_rows.append(format_latex_row(total_row))

    latex_footer = r"\hline" + "\n" + r"\end{tabularx}"

    # Helpful commented header telling the user which packages are required in their document preamble
    preamble_comment = (
        "% The following packages are required in your main LaTeX document preamble:\n"
        "% \\usepackage[table]{xcolor}  % for \\cellcolor and color models\\n"
        "% \\usepackage{colortbl}     % for table coloring (optional with xcolor)\\n"
        "% \\usepackage{tabularx}     % for tabularx and X column type\\n"
        "% \\usepackage{multirow}     % for multirow cells\\n"
        "% \\usepackage{array}        % for newcolumntype and array features\\n"
        "% If you want a self-contained file, run the script with --standalone.\n\n"
    )

    # Combine full table (fragment)
    latex_fragment = preamble_comment + latex_header + "\n".join(latex_rows) + "\n" + latex_footer

    # Optionally wrap into a minimal standalone LaTeX document
    if standalone:
        standalone_preamble = (
            r"\\documentclass{article}\n"
            r"\\usepackage[table]{xcolor}\n"
            r"\\usepackage{colortbl}\n"
            r"\\usepackage{tabularx}\n"
            r"\\usepackage{multirow}\n"
            r"\\usepackage{array}\n"
            r"\\usepackage[margin=1in]{geometry}\n"
            r"\\begin{document}\n"
        )
        standalone_footer = r"\\end{document}"
        latex_table = (
            standalone_preamble + latex_header + "\n".join(latex_rows) + "\n" + latex_footer + "\n" + standalone_footer
        )
    else:
        latex_table = latex_fragment

    # Save to file
    with open("combined_table.tex", "w", encoding="UTF-8") as f:
        f.write(latex_table)

    print("✅ LaTeX table saved to 'combined_table.tex'")


def main() -> None:
    """main method"""
    parser = argparse.ArgumentParser(description="Combine one or more CSV files into a LaTeX table.")
    parser.add_argument("--csv_files", nargs="+", help="CSV files to process (in order)")
    parser.add_argument(
        "--model-names",
        nargs="+",
        help="Optional model display names matching the order of csv_files (one name per file)",
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Generate a standalone .tex document (including preamble) instead of a fragment",
    )
    parser.add_argument(
        "--auto",
        help="Generates CSVs according to provided yaml",
    )
    args = parser.parse_args()

    if args.auto:
        tmp_dir = tempfile.mkdtemp(prefix="crs_eval_csvs", dir="/tmp")
        model_names, csv_files = generate_csvs(args.auto, tmp_dir)
    else:
        csv_files = args.csv_files

        # Validate provided files exist
        missing = [f for f in csv_files if not os.path.exists(f)]
        if missing:
            print(f"Error: the following files were not found: {', '.join(missing)}", file=sys.stderr)
            sys.exit(2)

        # Assign a model name to each CSV file for display
        # Priority: explicit --model-names (must match count) > default two names when two files > derive from filenames
        if args.model_names:
            if len(args.model_names) != len(csv_files):
                print("Error: --model-names must provide exactly one name per csv file.", file=sys.stderr)
                sys.exit(2)
            model_names = args.model_names
        else:
            default_two = ["gpt-oss", "Qwen"]
            if len(csv_files) == len(default_two):
                model_names = default_two
            else:
                model_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    run(model_names, csv_files, args.standalone)


if __name__ == "__main__":
    main()
