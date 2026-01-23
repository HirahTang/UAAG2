# Load Testing

This project uses [Locust](https://locust.io/) for load testing the API.

## Locust Test Files

The load test configuration is located in `tests/performancetests/locustfile.py`. Currently, it includes:

- **Health Check**: Tests the `/health` endpoint for basic availability and response time

## Running Load Tests Locally

1. Install dependencies:
   ```bash
   uv sync --locked --dev
   ```

2. Start the API locally (optional, for local testing):
   ```bash
   uv run uvicorn src.uaag2.api:app --host 0.0.0.0 --port 8000
   ```

3. Run Locust with the web UI:
   ```bash
   uv run locust -f tests/performancetests/locustfile.py --host http://localhost:8000
   ```
   Then open http://localhost:8089 in your browser.

4. Run Locust in headless mode:
   ```bash
   uv run locust -f tests/performancetests/locustfile.py \
     --host http://localhost:8000 \
     --users 10 \
     --spawn-rate 2 \
     --run-time 30s \
     --headless \
     --only-summary
   ```

## Automated Load Testing (CI/CD)

Load tests run automatically via GitHub Actions after each successful deployment to Cloud Run (on the `mlops` branch).

The workflow can also be triggered manually from the Actions tab by providing a custom API URL.

See `.github/workflows/load_test.yaml` for the workflow configuration.
