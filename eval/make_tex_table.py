"""Generate a latex table from the .json file produced by to inspect eval script"""

import argparse
import json
from pathlib import Path

TEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    """
    Escape special LaTeX characters in a string.
    """

    for char, replacement in TEX_REPLACEMENTS.items():
        s = s.replace(char, replacement)
    return s


def main() -> None:
    """main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=Path, help="json file with eval data")
    parser.add_argument("--with-tools", action="store_true", help="Include information about tool calls.")
    args = parser.parse_args()

    with open(args.data_file, "r+t", encoding="utf-8") as f:
        data: dict[str, dict[str, dict[str, int]]] = json.load(f)

    table = [r"\begin{table}[!pht]", r"\centering"]

    num_models = len(data.keys())

    table.append(r"\begin{tabular}{l" + num_models * 3 * "c" + "}")
    table.append(r"\toprule")
    table.append(r"\multirow{2}{*}{\textbf{Identifier}} &")

    for model_name in data.keys():
        table.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{tex_escape(model_name)}}}}} &")

    table[-1] = table[-1].replace("&", r"\\")

    c = 2
    for model_name in data.keys():
        table.append(rf"\cmidrule(lr){{{c}-{c+2}}}")
        c += 3

    for model_name in data.keys():
        table.append(r"& \textbf{Total} & \textbf{Vulnerable} & \textbf{Verified} ")
    table[-1] += r"\\"

    table.append(r"\midrule")

    zipped = {key: list(sub[key] for sub in data.values()) for key in next(iter(data.values()))}

    for instance_name, instance_results in zipped.items():
        table.append(f"{tex_escape(instance_name)} & ")

        for r in instance_results:
            table[-1] += f"{r['total']} & {r['marked_as_vulnerable']} & {r['reproducer_found']} & "

        table[-1] = table[-1][:-2]
        table[-1] += r"\\"

    if args.with_tools:
        table.append(r"\midrule")

        table.append(r"\textbf{Invalid tool name} & ")
        for instances in data.values():
            invalid_tool_name = 0
            for instance in instances.values():
                invalid_tool_name += instance["invalid_tool_name"]
            table.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{invalid_tool_name}}}}} &")

        table[-1] = table[-1].replace("&", r"\\")

        table.append(r"\textbf{Invalid tool parameter} & ")
        for instances in data.values():
            invalid_tool_params = 0
            for instance in instances.values():
                invalid_tool_params += instance["invalid_tool_params"]
            table.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{invalid_tool_params}}}}} &")

        table[-1] = table[-1].replace("&", r"\\")

        table.append(r"\textbf{No tool called} & ")
        for instances in data.values():
            no_tool_called = 0
            for instance in instances.values():
                no_tool_called += instance["no_tool_called"]
            table.append(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{no_tool_called}}}}} &")

        table[-1] = table[-1].replace("&", r"\\")

    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\caption{Model Comparison}")
    table.append(r"\end{table}")

    print("\n".join(table))


if __name__ == "__main__":
    main()
