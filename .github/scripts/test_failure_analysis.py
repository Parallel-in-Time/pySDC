#!/usr/bin/env python3
"""
Test script to validate the failure analysis logic.
This creates a mock scenario to test the scripts without actually running CI.
"""

import os
import sys
import json
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_error_extraction():
    """Test the error extraction function."""
    from analyze_failures import extract_error_summary
    
    # Mock log with various error patterns
    mock_log = """
2024-01-01 10:00:00 Running tests...
2024-01-01 10:00:01 Setting up environment
2024-01-01 10:00:05 ERROR: Test failed with exception
2024-01-01 10:00:05 Traceback (most recent call last):
  File "test_something.py", line 42, in test_function
    assert result == expected
AssertionError: 5 != 6
2024-01-01 10:00:10 FAILED test_something.py::test_function
2024-01-01 10:00:15 ModuleNotFoundError: No module named 'some_package'
2024-01-01 10:00:20 Continuing with other tests...
"""
    
    errors = extract_error_summary(mock_log, "test_job")
    
    print("✓ Error extraction test")
    print(f"  Found {len(errors)} error patterns")
    assert len(errors) > 0, "Should extract at least one error"
    
    # Check that we captured the traceback
    traceback_found = any('Traceback' in error for error in errors)
    assert traceback_found, "Should capture traceback"
    print("  ✓ Traceback captured")
    
    # Check that we captured other errors
    module_error_found = any('ModuleNotFoundError' in error for error in errors)
    assert module_error_found, "Should capture ModuleNotFoundError"
    print("  ✓ Module error captured")
    
    return True


def test_report_generation():
    """Test the report generation function."""
    from analyze_failures import generate_failure_report
    
    # Mock analysis data
    mock_analysis = {
        'total_jobs': 10,
        'failed_jobs': [
            {
                'name': 'test_job_1',
                'id': 12345,
                'started_at': '2024-01-01T10:00:00Z',
                'completed_at': '2024-01-01T10:05:00Z',
                'html_url': 'https://github.com/example/repo/actions/runs/123/jobs/12345',
                'errors': [
                    'ERROR: Something went wrong',
                    'Traceback (most recent call last):\n  File "test.py", line 1\n    raise Exception("test")'
                ]
            },
            {
                'name': 'test_job_2',
                'id': 12346,
                'started_at': '2024-01-01T10:00:00Z',
                'completed_at': '2024-01-01T10:06:00Z',
                'html_url': 'https://github.com/example/repo/actions/runs/123/jobs/12346',
                'errors': []
            }
        ],
        'timestamp': '2024-01-01T10:10:00Z'
    }
    
    run_url = 'https://github.com/example/repo/actions/runs/123'
    report = generate_failure_report(mock_analysis, run_url)
    
    print("✓ Report generation test")
    assert '# Automated Test Failure Analysis' in report, "Should have title"
    print("  ✓ Report has title")
    assert 'Total Jobs: 10' in report, "Should show total jobs"
    print("  ✓ Shows total jobs")
    assert 'Failed Jobs: 2' in report, "Should show failed job count"
    print("  ✓ Shows failed jobs count")
    assert 'test_job_1' in report, "Should list first failed job"
    assert 'test_job_2' in report, "Should list second failed job"
    print("  ✓ Lists all failed jobs")
    assert 'ERROR: Something went wrong' in report, "Should include error details"
    print("  ✓ Includes error details")
    
    return True


def test_pr_body_generation():
    """Test PR body generation."""
    from create_failure_pr import generate_pr_body
    
    # Create a temporary analysis file
    temp_dir = Path('/tmp/test_failure_analysis')
    temp_dir.mkdir(exist_ok=True)
    
    analysis_file = temp_dir / 'failure_analysis.json'
    mock_analysis = {
        'total_jobs': 15,
        'failed_jobs': [
            {'name': 'job1', 'id': 1},
            {'name': 'job2', 'id': 2},
            {'name': 'job3', 'id': 3}
        ],
        'timestamp': '2024-01-01T10:00:00Z'
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(mock_analysis, f)
    
    workflow_url = 'https://github.com/example/repo/actions/runs/123'
    branch_name = 'auto-fix/test-failure-20240101-100000'
    pr_body = generate_pr_body(workflow_url, str(analysis_file), branch_name)
    
    print("✓ PR body generation test")
    assert '🔴 Automated Test Failure Report' in pr_body, "Should have title"
    print("  ✓ Has title")
    assert workflow_url in pr_body, "Should include workflow URL"
    print("  ✓ Includes workflow URL")
    assert '3 out of 15' in pr_body, "Should show correct failure count"
    print("  ✓ Shows correct failure count")
    assert 'Weekly scheduled run' in pr_body, "Should mention it's a weekly run"
    print("  ✓ Mentions weekly run")
    assert branch_name in pr_body, "Should include branch name in instructions"
    print("  ✓ Includes branch name")
    
    # Clean up
    analysis_file.unlink()
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Automated Failure Analysis Scripts")
    print("=" * 60)
    print()
    
    tests = [
        ("Error Extraction", test_error_extraction),
        ("Report Generation", test_report_generation),
        ("PR Body Generation", test_pr_body_generation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            print("-" * 60)
            test_func()
            passed += 1
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
