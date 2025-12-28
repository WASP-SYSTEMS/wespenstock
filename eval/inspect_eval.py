"""Module to inspect CRS evlaluation results"""

import argparse
import getpass
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

import pandas as pd
from dateutil import parser as dateparser
from prettytable import PrettyTable

from eval.parse_logs import get_test_series_data

# define identifier to CPV mapping
IDENTIFIER_TO_CPV = {
    "0dbd46415432759475e6e4bb5adfaada6fb7d506": "02",
    "316d57f895c4c915c5ce3af8b09972d47dd9984e": "13",
    "348b50dbb52e7d6faad7d75ce9331dd9860131c4": "12",
    "9c5e32dcd9779c4cfe48c5377d0af8adc52a2be9": "14",
    "a2f5fad3ef16615ed23d21264560748cdc21a385": "11",
    "b101d59b3dda654dee1deabc34816e2ca7c96d38": "05",
    "b6c0a37554e300aa230ea2b8d7fe53dd8604f602": "17",
    "b9d6a2caf41565fb05c010ad0c8d2fd6bd3c4c42": "04",
    "c502a1695c0e9d0345101a5f2a99ee0e3c890a4d": "03",
    "cc4b16fc10dcc579d5f697f3ff70c390b5e7c7d2": "09",
    "cf6f5b1d4d85c98b4e2e2fb6f694f996d944851a": "08",
    "d030af5eb4c64470c8fd5a87a8f6aae547580aa3": "01",
    "dcf9f055bf1863555869493d5ac5944b5327f128": "10",
    "ef970a54395324307fffd11ab37266479ac37d4c": "15",
    "ngx_http_validate_from": "01",
    "ngx_http_auth_basic_user": "02",
    "ngx_http_trace_handler": "03",
    "ngx_http_set_browser_cookie": "04",
    "ngx_get_con_his": "05",
    "ngx_http_get_last_ip_variable": "05",
    "ngx_mail_pop3_user": "08",
    "ngx_black_list_insert": "09",
    "ngx_black_list_remove": "09",
    "ngx_is_ip_banned": "09",
    "ngx_http_process_prefer": "10",
    "ngx_init_cycle": "11",
    "ngx_sendfile_r": "12",
    "ngx_mail_pop3_pass": "13",
    "ngx_mail_pop3_logs": "13",
    "ngx_http_script_regex_end_code": "14",
    "ngx_http_userid_get_uid": "15",
    "ngx_mail_smtp_noop": "17",
}


class TestSeriesRow(TypedDict):
    """
    Typing for evaluate_test_series
    """

    identifier: str
    total: int
    successful: int
    secure: Optional[int]
    vulnerable: Optional[int]


def evaluate_test_series(
    test_series: Path,
    with_predictions: bool = False,
) -> List[TestSeriesRow]:
    """
    Parses test series output and generates a list of results per identifier (and total) to be used
    to generate a table
    """
    results: List[TestSeriesRow] = []
    test_series_data = get_test_series_data(test_series) if with_predictions else []

    for identifier_path in sorted(test_series.iterdir()):
        if not identifier_path.is_dir():
            continue

        total_samples = 0
        successful_samples = 0
        for sample_path in identifier_path.iterdir():
            if sample_path.is_dir():
                total_samples += 1
                if any(f.name.startswith("pov_success_") for f in sample_path.iterdir() if f.is_file()):
                    successful_samples += 1

        secure = None
        vulnerable = None
        if with_predictions and test_series_data:
            for cur_test_series in test_series_data:
                if cur_test_series.folder != test_series:
                    continue
                for analyzer_info in cur_test_series.analyzer_info:
                    if analyzer_info.file == Path(f"{identifier_path.name}.json"):
                        secure = analyzer_info.secure
                        vulnerable = analyzer_info.vulnerable
                        break
                break

        results.append(
            {
                "identifier": identifier_path.name,
                "total": total_samples,
                "successful": successful_samples,
                "secure": secure,
                "vulnerable": vulnerable,
            }
        )

    return results


def generate_results_table(results: List[TestSeriesRow]) -> PrettyTable:
    """
    Print a pretty table with CPV - Identifier - Total Samples - Successful Samples
    If results include predictions, also show Vulnerable/Secure.
    """
    table_cpv = PrettyTable()
    samples = 0
    samples_success = 0
    identifier_count = 0
    secure_count = 0
    vulnerable_count = 0

    # Detect if predictions are present (check first element keys)
    with_predictions = (
        len(results) > 0 and results[0].get("secure") is not None and results[0].get("vulnerable") is not None
    )
    if with_predictions:
        table_cpv.field_names = ["CPV", "Identifier", "Total Samples", "Successful Samples", "Vulnerable", "Secure"]
        for row in results:
            identifier = row["identifier"]
            total = row["total"]
            successful = row["successful"]
            secure = row["secure"] or 0
            vulnerable = row["vulnerable"] or 0

            samples += total
            samples_success += successful
            identifier_count += 1
            secure_count += secure
            vulnerable_count += vulnerable

            table_cpv.add_row(
                [IDENTIFIER_TO_CPV.get(identifier, "N/A"), identifier, total, successful, vulnerable, secure]
            )

        table_cpv.sortby = "CPV"
        table_cpv.add_row(
            ["TOTAL", identifier_count, samples, samples_success, vulnerable_count, secure_count],
            divider=True,
        )

    else:
        table_cpv.field_names = ["CPV", "Identifier", "Total Samples", "Successful Samples"]
        for row in results:
            identifier = row["identifier"]
            total = row["total"]
            successful = row["successful"]

            samples += total
            samples_success += successful
            identifier_count += 1

            table_cpv.add_row([IDENTIFIER_TO_CPV.get(identifier, "N/A"), identifier, total, successful])

        table_cpv.sortby = "CPV"
        table_cpv.add_row(
            ["TOTAL", identifier_count, samples, samples_success],
            divider=True,
        )

    return table_cpv


# an empty series list will be fixed first thing in this function so we do not need to force one to be provided
# pylint: disable=dangerous-default-value
def merge_pretty_tables_to_pandas(tables: List[PrettyTable], series_names: List[str] = []) -> pd.DataFrame:
    """
    Merge multiple PrettyTables into a single pandas DataFrame
    with multi-index columns for easy comparison.

    Args:
        tables (list): list of PrettyTable objects
        series_names (list): optional list of names for each table
                             (default: Series1, Series2, ...)

    Returns:
        pd.DataFrame: merged DataFrame with MultiIndex columns
    """
    if not series_names:
        series_names = [f"Series{i+1}" for i in range(len(tables))]

    dfs = []
    for table, name in zip(tables, series_names):
        # Convert PrettyTable to DataFrame
        df = pd.DataFrame(table.rows, columns=table.field_names)

        # Ensure TOTAL row uses "-" for identifier (avoids multi-index confusion)
        df["Identifier"] = df["Identifier"].where(df["CPV"] != "TOTAL", "-")

        # Base mandatory columns
        base_cols = ["CPV", "Identifier", "Total Samples", "Successful Samples"]

        # Detect predictions
        has_predictions = "Secure" in df.columns and "Vulnerable" in df.columns

        # Enforce column order depending on schema
        if has_predictions:
            df = df[base_cols + ["Secure", "Vulnerable"]]
        else:
            df = df[base_cols]

        # Rename for multi-series comparison
        rename_map = {
            "Total Samples": (name, "total"),
            "Successful Samples": (name, "success"),
        }
        if has_predictions:
            rename_map.update(
                {
                    "Secure": (name, "secure"),
                    "Vulnerable": (name, "vulnerable"),
                }
            )

        df = df.rename(columns=rename_map)
        df = df.set_index(["CPV", "Identifier"])
        dfs.append(df)

    # Merge all DataFrames on (CPV, Identifier)
    merged = pd.concat(dfs, axis=1).sort_index()

    # Fill NaNs with 0 and cast numeric columns
    merged = merged.fillna(0)
    for col in merged.columns:
        # only convert numeric-like columns
        if merged[col].dtype in (float, int):
            merged[col] = merged[col].astype(int)

    # Ensure proper MultiIndex for all columns
    merged.columns = pd.MultiIndex.from_tuples([col if isinstance(col, tuple) else ("", col) for col in merged.columns])

    return merged


def parse_log_line(line: str) -> Dict[str, str]:
    """parses a logline and returns key value pairs"""
    pattern = r'(\w+)=(".*?"|\S+)'
    matches = re.findall(pattern, line)
    return {key: value.strip('"') for key, value in matches}


def find_last_user_log(filepath: Path, current_user: str, series_type: str = "nginx") -> Dict[str, str] | None:
    """finds last line in the log created by the current user for given test series type (nginx or 0days)"""
    last_match = None

    with open(filepath, encoding="UTF-8") as f:
        for line in f:
            data = parse_log_line(line)
            if data.get("user") == current_user and data.get("type") == series_type:
                last_match = data

    return last_match


def format_timedelta(timestamp_str: str) -> str:
    """adds time difference to a timestamp"""
    try:
        timestamp = dateparser.parse(timestamp_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - timestamp

        total_minutes = int(delta.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        if delta.days > 0:
            return f"{timestamp_str} (that's {delta.days}d {hours}h {minutes}m ago)"
        return f"{timestamp_str} (that's {hours}h {minutes}m ago)"
    except Exception:  # pylint: disable=broad-exception-caught
        return f"{timestamp_str} (invalid timestamp)"


def main() -> None:
    """
    Parse test series as input and print its result as pretty table
    """
    parser = argparse.ArgumentParser(description="Evaluate test series for pov_successful_TIMESTAMP files.")
    parser.add_argument("test_series_dir", nargs="?", type=Path, help="Path to the test_series directory.")
    parser.add_argument("--last", action="store_true", help="Inspects your last run according to /data/CRSLOG.")
    parser.add_argument("--no-predictions", action="store_true", help="Do not show Analyzer predictions in table")
    parser.add_argument("--user", type=str, help="Inspect another users last run according to /data/CRSLOG")
    parser.add_argument("--csv", type=str, help="Output results to CSV file.")
    args = parser.parse_args()

    if args.user:
        args.last = True

    if not args.last and not args.test_series_dir:
        parser.error("You must specify either a test_series_dir or use the --last/--user flag.")

    if args.last:
        if args.user:
            user = args.user
        else:
            user = getpass.getuser()
        last_user_log = find_last_user_log(Path("/data/CRSLOG"), user, "nginx")
        if last_user_log:
            print("Last matching log entry:")
            for key, value in last_user_log.items():
                if key == "timestamp":
                    print(f"{key}: {format_timedelta(value)}")
                else:
                    print(f"{key}: {value}")
            test_series_dir = Path(last_user_log["eval_results_path"])
        else:
            print("No matching log entry found for current user.")
            return
    else:
        test_series_dir = args.test_series_dir

    results = evaluate_test_series(test_series_dir, (not args.no_predictions))
    table = generate_results_table(results)
    print(table)

    if args.csv:
        df = pd.DataFrame(table.rows, columns=table.field_names)
        df.to_csv(args.csv, index=False)
        print(f"Results saved to {args.csv}")


if __name__ == "__main__":
    main()
