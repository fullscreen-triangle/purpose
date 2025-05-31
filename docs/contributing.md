---
layout: default
title: Contributing
nav_order: 6
---

# Contributing to Purpose

We love your input! We want to make contributing to Purpose as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Pull Request Process

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the documentation with any new features or changes.
3. The PR will be merged once you have the sign-off of at least one other developer.

## Any Contributions You Make Will Be Under the Software License

In short, when you submit code changes, your submissions are understood to be under the same [LICENSE](../LICENSE) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report Bugs Using GitHub's Issue Tracker

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/purpose/issues/new); it's that easy!

## Write Bug Reports with Detail, Background, and Sample Code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Use a Consistent Coding Style

* 4 spaces for indentation rather than tabs
* 80 character line length
* Run `black` for Python code formatting
* Follow PEP 8 guidelines

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References

This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md).

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/purpose.git
   cd purpose
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

```bash
pytest tests/
```

## Code Quality Checks

Before submitting a PR, please run:

```bash
# Format code
black .

# Check style
flake8

# Type checking
mypy .

# Run tests
pytest
```

## Documentation

We use Jekyll for our documentation. To preview documentation locally:

1. Install Ruby and Bundler
2. Run:
   ```bash
   cd docs
   bundle install
   bundle exec jekyll serve
   ```

## Getting Help

If you need help with contributing:

1. Check the documentation
2. Join our community discussions
3. Ask questions in GitHub issues
4. Contact the maintainers

Thank you for contributing to Purpose! 