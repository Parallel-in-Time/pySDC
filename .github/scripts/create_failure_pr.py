#!/usr/bin/env python3
"""
Create a Pull Request for test failure analysis and fixes.
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any


def get_github_headers(token: str) -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    return {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }


def create_pull_request(
    repo: str,
    token: str,
    head_branch: str,
    base_branch: str,
    title: str,
    body: str
) -> Dict[str, Any]:
    """Create a pull request using GitHub API."""
    url = f'https://api.github.com/repos/{repo}/pulls'
    headers = get_github_headers(token)
    
    data = {
        'title': title,
        'body': body,
        'head': head_branch,
        'base': base_branch
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    return response.json()


def generate_pr_body(workflow_run_url: str, analysis_file: str, head_branch: str) -> str:
    """Generate the pull request body.
    
    Args:
        workflow_run_url: URL to the failed workflow run
        analysis_file: Path to the failure analysis JSON file
        head_branch: Name of the branch containing the analysis (required)
    """
    # Try to load the analysis
    try:
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        
        failed_count = len(analysis.get('failed_jobs', []))
        total_count = analysis.get('total_jobs', 0)
    except Exception:
        failed_count = "Unknown"
        total_count = "Unknown"
    
    body = f"""## 🔴 Automated Test Failure Report

This PR was automatically created in response to test failures in the weekly CI run.

### Summary
- **Workflow Run:** {workflow_run_url}
- **Failed Jobs:** {failed_count} out of {total_count}
- **Trigger:** Weekly scheduled run (Monday morning)

### What This PR Contains

This PR includes an automated analysis of the test failures. The detailed report can be found in the committed `failure_analysis.md` file.

### Next Steps

1. **Review the Analysis:** Check the `failure_analysis.md` file for detailed error information
2. **Investigate Root Cause:** Review the workflow logs and error messages
3. **Apply Fixes:** If you identify the issue, commit fixes to this branch
4. **Test Locally:** Reproduce and verify the fix before merging
5. **Update CI:** Ensure the fix resolves the weekly test failures

### How to Fix Issues

You can push commits directly to this branch:

```bash
git fetch origin
git checkout {head_branch}
# Make your changes
git add .
git commit -m "Fix: describe your fix"
git push origin {head_branch}
```

### Alternative Actions

- If this is a **transient failure**, you can close this PR
- If this requires **more investigation**, convert this PR to an issue
- If this is a **known issue**, link it to existing issues/PRs

---

**Note:** This is an automated PR. Please review carefully before merging.
"""
    
    return body


def main():
    """Main function to create the PR."""
    parser = argparse.ArgumentParser(description='Create PR for test failures')
    parser.add_argument('--branch', required=True, help='Branch name for the PR')
    parser.add_argument('--workflow-run-id', required=True, help='Workflow run ID')
    parser.add_argument('--workflow-run-url', required=True, help='Workflow run URL')
    parser.add_argument('--base', default='master', help='Base branch (default: master)')
    
    args = parser.parse_args()
    
    token = os.environ.get('GITHUB_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY')
    
    if not token or not repo:
        print("Error: Missing GITHUB_TOKEN or GITHUB_REPOSITORY environment variables")
        sys.exit(1)
    
    analysis_file = '.github/failure_analysis.json'
    
    # Check if analysis file exists
    if not os.path.exists(analysis_file):
        print(f"Warning: Analysis file {analysis_file} not found")
        print("Creating PR without detailed analysis")
    
    # Generate PR body
    pr_body = generate_pr_body(args.workflow_run_url, analysis_file, args.branch)
    
    # Create PR title
    pr_title = f"🔴 Auto-fix: Weekly test failures ({args.workflow_run_id})"
    
    try:
        pr = create_pull_request(
            repo=repo,
            token=token,
            head_branch=args.branch,
            base_branch=args.base,
            title=pr_title,
            body=pr_body
        )
        
        print(f"✅ Pull request created successfully!")
        print(f"PR Number: #{pr['number']}")
        print(f"PR URL: {pr['html_url']}")
        
        # Add labels to the PR
        try:
            labels_url = f"https://api.github.com/repos/{repo}/issues/{pr['number']}/labels"
            headers = get_github_headers(token)
            labels_data = {'labels': ['automated', 'test-failure', 'needs-investigation']}
            requests.post(labels_url, headers=headers, json=labels_data)
            print("✅ Labels added to PR")
        except Exception as e:
            print(f"Warning: Could not add labels: {e}")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            # PR might already exist or no changes
            print("⚠️  Could not create PR - it might already exist or there are no changes")
            print(f"Response: {e.response.text}")
        else:
            print(f"❌ Error creating PR: {e}")
            print(f"Response: {e.response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error creating PR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
