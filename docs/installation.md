---
layout: default
title: Installation
nav_order: 2
---

# Installation Guide

## Prerequisites

Before installing Purpose, ensure you have the following prerequisites:

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation Methods

### Using pip

The simplest way to install Purpose is using pip:

```bash
pip install purpose
```

### From Source

To install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/purpose.git
   cd purpose
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install in development mode:
   ```bash
   pip install -e .
   ```

## Verifying Installation

To verify that Purpose is installed correctly, run:

```python
import purpose
print(purpose.__version__)
```

## Dependencies

The main dependencies are listed in `requirements.txt` and include:

```
# List key dependencies from requirements.txt
```

## Troubleshooting

If you encounter any issues during installation:

1. Ensure all prerequisites are met
2. Check your Python version
3. Verify your virtual environment is activated
4. Make sure all dependencies are properly installed

For more help, please [open an issue](https://github.com/yourusername/purpose/issues) on our GitHub repository. 