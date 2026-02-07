#!/usr/bin/env python3
"""
Analyze GitHub Actions workflow failures and generate a detailed report.
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Constants for error processing
ERROR_UNIQUENESS_KEY_LENGTH = 100  # Characters to use for error deduplication
MAX_ERROR_LENGTH = 500  # Maximum characters to include per error in report
MAX_ERRORS_PER_JOB = 10  # Maximum unique errors to extract per job


def get_github_headers(token: str) -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    return {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }


def get_workflow_run_jobs(repo: str, run_id: str, token: str) -> List[Dict[str, Any]]:
    """Fetch all jobs for a workflow run."""
    url = f'https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs'
    headers = get_github_headers(token)
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return response.json().get('jobs', [])


def get_job_logs(repo: str, job_id: int, token: str) -> Optional[str]:
    """Fetch logs for a specific job."""
    url = f'https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs'
    headers = get_github_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Warning: Could not fetch logs for job {job_id}: {e}")
        return None


def extract_error_summary(logs: str, job_name: str) -> List[str]:
    """Extract error messages and failures from job logs."""
    errors = []
    lines = logs.split('\n') if logs else []
    
    # Look for common error patterns
    error_patterns = [
        'ERROR:',
        'FAILED',
        'Error:',
        'error:',
        'Traceback (most recent call last):',
        'AssertionError',
        'ModuleNotFoundError',
        'ImportError',
        'FAIL:',
        'pytest failed',
        'Test failed',
    ]
    
    in_traceback = False
    traceback_lines = []
    
    for i, line in enumerate(lines):
        # Capture tracebacks
        if 'Traceback (most recent call last):' in line:
            in_traceback = True
            traceback_lines = [line]
        elif in_traceback:
            traceback_lines.append(line)
            # End of traceback
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                errors.append('\n'.join(traceback_lines))
                in_traceback = False
                traceback_lines = []
        else:
            # Check for error patterns
            for pattern in error_patterns:
                if pattern in line:
                    # Add context (previous and next lines)
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context = '\n'.join(lines[context_start:context_end])
                    errors.append(context)
                    break
    
    # Limit to unique errors and first MAX_ERRORS_PER_JOB
    unique_errors = []
    seen = set()
    for error in errors:
        error_key = error[:ERROR_UNIQUENESS_KEY_LENGTH]  # Use first N chars as key
        if error_key not in seen:
            seen.add(error_key)
            unique_errors.append(error)
            if len(unique_errors) >= MAX_ERRORS_PER_JOB:
                break
    
    return unique_errors


def analyze_failures(repo: str, run_id: str, token: str) -> Dict[str, Any]:
    """Analyze all failures in a workflow run."""
    jobs = get_workflow_run_jobs(repo, run_id, token)
    
    analysis = {
        'total_jobs': len(jobs),
        'failed_jobs': [],
        'error_summary': [],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    for job in jobs:
        if job['conclusion'] == 'failure':
            job_info = {
                'name': job['name'],
                'id': job['id'],
                'started_at': job['started_at'],
                'completed_at': job['completed_at'],
                'html_url': job['html_url'],
                'errors': []
            }
            
            # Fetch and analyze logs
            logs = get_job_logs(repo, job['id'], token)
            if logs:
                errors = extract_error_summary(logs, job['name'])
                job_info['errors'] = errors
            
            analysis['failed_jobs'].append(job_info)
    
    return analysis


def generate_failure_report(analysis: Dict[str, Any], run_url: str) -> str:
    """Generate a markdown report of the failures."""
    report = [
        "# Automated Test Failure Analysis",
        "",
        f"**Generated:** {analysis['timestamp']}",
        f"**Workflow Run:** {run_url}",
        "",
        "## Summary",
        "",
        f"- Total Jobs: {analysis['total_jobs']}",
        f"- Failed Jobs: {len(analysis['failed_jobs'])}",
        "",
    ]
    
    if not analysis['failed_jobs']:
        report.append("✅ No failed jobs detected (this might be a transient issue).")
        return '\n'.join(report)
    
    report.extend([
        "## Failed Jobs",
        "",
    ])
    
    for i, job in enumerate(analysis['failed_jobs'], 1):
        report.extend([
            f"### {i}. {job['name']}",
            "",
            f"- **Job ID:** {job['id']}",
            f"- **Started:** {job['started_at']}",
            f"- **Completed:** {job['completed_at']}",
            f"- **Logs:** [View Job Logs]({job['html_url']})",
            "",
        ])
        
        if job['errors']:
            report.extend([
                "#### Error Details",
                "",
            ])
            for j, error in enumerate(job['errors'][:5], 1):  # Limit to 5 errors per job
                report.extend([
                    f"**Error {j}:**",
                    "```",
                    error[:MAX_ERROR_LENGTH],  # Limit error length
                    "```",
                    "",
                ])
        else:
            report.append("No specific error messages extracted. Check job logs for details.")
            report.append("")
    
    report.extend([
        "## Recommended Actions",
        "",
        "1. Review the error messages above",
        "2. Check if this is a known issue in recent commits",
        "3. Review the full logs linked above for complete context",
        "4. Consider if this is related to:",
        "   - Dependency updates (check recent dependency changes)",
        "   - Environment configuration issues",
        "   - Test infrastructure problems",
        "   - Flaky tests that need to be fixed",
        "5. If needed, manually investigate and apply fixes to this PR",
        "",
        "## How to Use This PR",
        "",
        "This PR was automatically created to help investigate test failures. You can:",
        "",
        "- Use this PR to track the investigation",
        "- Add commits with fixes directly to this branch",
        "- Close this PR if the issue is resolved elsewhere",
        "- Convert this to an issue if it needs more discussion",
        "",
    ])
    
    return '\n'.join(report)


def main():
    """Main function to analyze failures and create report."""
    token = os.environ.get('GITHUB_TOKEN')
    run_id = os.environ.get('WORKFLOW_RUN_ID')
    repo = os.environ.get('REPOSITORY')
    
    if not all([token, run_id, repo]):
        print("Error: Missing required environment variables")
        print("Required: GITHUB_TOKEN, WORKFLOW_RUN_ID, REPOSITORY")
        sys.exit(1)
    
    print(f"Analyzing failures for workflow run {run_id} in {repo}")
    
    try:
        # Analyze failures
        analysis = analyze_failures(repo, run_id, token)
        
        # Get workflow run URL
        run_url = f"https://github.com/{repo}/actions/runs/{run_id}"
        
        # Generate report
        report = generate_failure_report(analysis, run_url)
        
        # Save report
        os.makedirs('.github', exist_ok=True)
        with open('.github/failure_analysis.md', 'w') as f:
            f.write(report)
        
        print(f"✅ Analysis complete. Found {len(analysis['failed_jobs'])} failed jobs.")
        print(f"Report saved to .github/failure_analysis.md")
        
        # Save analysis as JSON for the PR script
        with open('.github/failure_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
    except Exception as e:
        print(f"❌ Error analyzing failures: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
