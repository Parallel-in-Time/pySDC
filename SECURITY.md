# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 5.x     | :white_check_mark: |
| < 5.0   | :x:                |

We recommend using the latest version of pySDC to ensure you have the latest security updates.

## Reporting a Vulnerability

We take the security of pySDC seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

- **r.speck@fz-juelich.de** (Robert Speck, Project Maintainer)

You should receive a response within 5 business days. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information in your report (as much as you can provide):

- Type of issue (e.g., arbitrary code execution, unsafe deserialization, path traversal, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Security Update Policy

Security updates will be released as soon as possible after a vulnerability is confirmed and a fix is available. Updates will be announced through:

- GitHub Security Advisories
- Release notes in the [CHANGELOG](./CHANGELOG.md)
- The project's GitHub Releases page

## Dependencies

pySDC uses several third-party dependencies. We monitor our dependencies for known security vulnerabilities and update them as needed. If you discover a security issue in one of our dependencies, please also report it to the respective maintainers of that dependency.

## Comments on This Policy

If you have suggestions on how this process could be improved, please submit a pull request or open an issue to discuss.
