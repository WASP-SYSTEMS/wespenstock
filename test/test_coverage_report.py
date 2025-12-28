#!/usr/bin/env python3
"""
Unit tests for the LLVM coverage parser.

Parts of this are vibed.
"""

from pathlib import Path

import pytest

from crs.agents.pov_builder.coverage.coverage_report import CoverageReport


# TODO: test source code matching, I'm too lazy to init a file structure for it rn
class TestCoverageParser:
    """Test cases for the coverage parser."""

    def test_accessor(self) -> None:
        """test if the getitem works as expected"""
        simple_coverage = """SF:/test/file.c
FN:1,test_function
FNDA:5,test_function
FNF:1
FNH:1
DA:1,5
DA:2,3
DA:3,0
LF:3
LH:2
end_of_record"""

        result = CoverageReport.parse_coverage(simple_coverage)
        assert result.functions["test_function"] == result["test_function"]

    def test_parse_simple_coverage_content(self) -> None:
        """Test parsing simple coverage content."""
        simple_coverage = """SF:/test/file.c
FN:1,test_function
FNDA:5,test_function
FNF:1
FNH:1
DA:1,5
DA:2,3
DA:3,0
LF:3
LH:2
end_of_record"""

        result = CoverageReport.parse_coverage(simple_coverage)

        assert isinstance(result, CoverageReport)
        assert len(result.functions) == 1
        assert "test_function" in result.functions
        assert result.functions["test_function"].source_file == Path("test/file.c")

        func_coverage = result.functions["test_function"]
        assert len(func_coverage.lines) == 3
        assert func_coverage.lines[1].executions == 5
        assert func_coverage.lines[2].executions == 3
        assert func_coverage.lines[3].executions == 0

    def test_parse_coverage_with_branches(self) -> None:
        """Test parsing coverage content with branch information."""
        coverage_with_branches = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
DA:2,1
BRDA:2,0,0,0
BRDA:2,0,1,1
BRF:2
BRH:1
LF:2
LH:2
end_of_record"""

        result = CoverageReport.parse_coverage(coverage_with_branches)

        func_coverage = result.functions["test_function"]
        assert len(func_coverage.branches) == 1
        assert 2 in func_coverage.branches

        branch_info = func_coverage.branches[2][0]
        assert branch_info.true == 0
        assert branch_info.false == 1

    def test_parse_multiple_functions(self) -> None:
        """Test parsing coverage with multiple functions."""
        multi_function_coverage = """SF:/test/file.c
FN:1,func1
FN:10,func2
FNDA:2,func1
FNDA:1,func2
FNF:2
FNH:2
DA:1,2
DA:2,2
DA:10,1
DA:11,1
LF:4
LH:4
end_of_record"""

        result = CoverageReport.parse_coverage(multi_function_coverage)

        assert len(result.functions) == 2
        assert "func1" in result.functions
        assert "func2" in result.functions

        assert result.functions["func1"].lines[1].executions == 2
        assert result.functions["func2"].lines[10].executions == 1

    def test_parse_multiple_source_files(self) -> None:
        """Test parsing coverage with multiple source files."""
        multi_file_coverage = """SF:/file1.c
FN:1,func1
FNDA:1,func1
FNF:1
FNH:1
DA:1,1
LF:1
LH:1
end_of_record
SF:/file2.c
FN:1,func2
FNDA:1,func2
FNF:1
FNH:1
DA:1,1
LF:1
LH:1
end_of_record"""

        result = CoverageReport.parse_coverage(multi_file_coverage)

        assert len(result.functions) == 2
        assert "func1" in result.functions
        assert "func2" in result.functions

    def test_parse_complex_branch_coverage(self) -> None:
        """Test parsing complex branch coverage scenarios."""
        complex_branch_coverage = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
DA:2,1
DA:3,1
BRDA:2,0,0,0
BRDA:2,0,1,1
BRDA:3,0,0,1
BRDA:3,0,1,0
BRDA:3,1,0,1
BRDA:3,1,1,0
BRF:6
BRH:3
LF:3
LH:3
end_of_record"""

        result = CoverageReport.parse_coverage(complex_branch_coverage)

        func_coverage = result.functions["test_function"]

        # Line 2 has one branch condition
        assert 2 in func_coverage.branches
        assert len(func_coverage.branches[2]) == 1

        # Line 3 has two branch conditions
        assert 3 in func_coverage.branches
        assert len(func_coverage.branches[3]) == 2

    def test_parse_coverage_from_file(self, tmp_path: Path) -> None:
        """Test parsing coverage from a file."""
        coverage_content = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
LF:1
LH:1
end_of_record"""

        test_file = tmp_path / "test_coverage.txt"
        test_file.write_text(coverage_content)

        result = CoverageReport.parse_coverage_from_file(test_file)

        assert isinstance(result, CoverageReport)
        assert len(result.functions) == 1
        assert "test_function" in result.functions

    def test_parse_coverage_from_string(self) -> None:
        """Test parsing coverage from a string."""
        coverage_content = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
LF:1
LH:1
end_of_record"""

        result = CoverageReport.parse_coverage(coverage_content)

        assert isinstance(result, CoverageReport)
        assert len(result.functions) == 1

    def test_parse_empty_content(self) -> None:
        """Test parsing empty coverage content."""
        result = CoverageReport.parse_coverage("")

        assert isinstance(result, CoverageReport)
        assert len(result.functions) == 0

    def test_parse_content_with_only_whitespace(self) -> None:
        """Test parsing content with only whitespace."""
        result = CoverageReport.parse_coverage("   \n  \t  \n  ")

        assert isinstance(result, CoverageReport)
        assert len(result.functions) == 0

    def test_parse_incomplete_section(self) -> None:
        """Test parsing incomplete coverage section."""
        incomplete_coverage = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
DA:1,1"""
        with pytest.raises(FileNotFoundError):
            CoverageReport.parse_coverage_from_file(incomplete_coverage)

    def test_parse_with_checksum_in_da(self) -> None:
        """Test parsing DA lines with optional checksum."""
        coverage_with_checksum = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1,abc123
LF:1
LH:1
end_of_record"""
        result = CoverageReport.parse_coverage(coverage_with_checksum)

        func_coverage = result.functions["test_function"]
        assert func_coverage.lines[1].executions == 1

    def test_parse_real_coverage_data(self) -> None:
        """Test parsing the actual coverage data from cov.txt."""
        # Read the actual test data
        test_data_path = Path(__file__).parent / "data" / "cov.txt"

        if test_data_path.exists():
            result = CoverageReport.parse_coverage_from_file(test_data_path)

            assert isinstance(result, CoverageReport)
            assert len(result.functions) == 8

            # Check specific functions exist
            expected_functions = [
                "LLVMFuzzerTestOneInput",
                "parse_request",
                "handle_request",
                "respond",
                "free_request",
                "do_stuff",
                "sha1",
                "sha1_equal",
            ]

            for func_name in expected_functions:
                assert func_name in result.functions

            # Check some specific coverage data
            llvm_func = result.functions["LLVMFuzzerTestOneInput"]
            assert len(llvm_func.lines) == 6
            assert llvm_func.lines[9].executions == 1

            # Check branch coverage
            assert len(llvm_func.branches) == 2
            assert 10 in llvm_func.branches
            assert 11 in llvm_func.branches

    def test_file_not_found_error(self) -> None:
        """Test that appropriate error is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            CoverageReport.parse_coverage_from_file("/non/existent/file.txt")

    def test_coverage_statistics_parsing(self) -> None:
        """Test that coverage statistics are parsed correctly."""
        coverage_with_stats = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
DA:2,0
DA:3,1
BRDA:2,0,0,0
BRDA:2,0,1,1
BRF:2
BRH:1
LF:3
LH:2
end_of_record"""

        result = CoverageReport.parse_coverage(coverage_with_stats)

        func_coverage = result.functions["test_function"]

        # Check line coverage
        assert func_coverage.lines[1].executions == 1
        assert func_coverage.lines[2].executions == 0
        assert func_coverage.lines[3].executions == 1

        # Check branch coverage
        assert 2 in func_coverage.branches
        branch_info = func_coverage.branches[2][0]
        assert branch_info.true == 0
        assert branch_info.false == 1

    def test_coverage_branch_not_executed_parsing(self) -> None:
        """Test that coverage statistics are parsed correctly."""
        coverage_with_stats = """SF:/test/file.c
FN:1,test_function
FNDA:1,test_function
FNF:1
FNH:1
DA:1,1
DA:2,0
DA:3,1
BRDA:2,0,0,-
BRDA:2,0,1,-
BRF:2
BRH:1
LF:3
LH:2
end_of_record"""

        result = CoverageReport.parse_coverage(coverage_with_stats)

        func_coverage = result.functions["test_function"]

        # Check line coverage
        assert func_coverage.lines[1].executions == 1
        assert func_coverage.lines[2].executions == 0
        assert func_coverage.lines[3].executions == 1

        # Check branch coverage
        assert 2 in func_coverage.branches
        branch_info = func_coverage.branches[2][0]
        assert branch_info.true == 0
        assert branch_info.false == 0


if __name__ == "__main__":
    pytest.main([__file__])
