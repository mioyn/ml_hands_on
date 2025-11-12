## Using UV for Dependency Management and Execution

This project uses [UV](https://github.com/astral-sh/uv) for managing Python dependencies and running scripts efficiently.

### Common UV Commands

- **Sync dependencies (install from `requirements.txt`):**
    ```bash
    uv sync
    ```

- **Add a new dependency (e.g., OpenCV):**
    ```bash
    uv add opencv-python
    uv sync
    ```

- **Run a Python script:**
    ```bash
    uv run opencvdemo.py
    ```

- **Start a Python REPL:**
    ```bash
    uv run python
    ```

For more information, see the [UV documentation](https://github.com/astral-sh/uv).

