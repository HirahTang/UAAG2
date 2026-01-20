# Performance Tests

This directory contains performance and load tests for the UAAG2 MNIST API, following the DTU MLOps course module M24 (API Testing & Load Testing).

## Contents

- **`locustfile.py`**: Locust load testing scenarios for the API
- **`test_model.py`**: Performance tests for model inference speed

## Load Testing with Locust

### Prerequisites

Ensure you have installed the required dependencies:

```bash
# Using uv (recommended for this project)
uv sync --group dev

# Or using pip
pip install locust httpx
```

### Running Load Tests

#### 1. Start the API Server

First, start the API server locally:

```bash
# Using uvicorn directly
uvicorn uaag2.api:app --reload

# Or using uv
uv run uvicorn uaag2.api:app --reload
```

The API will be available at `http://localhost:8000`.

#### 2. Run Locust Tests

**Option A: Web UI Mode (Recommended for exploration)**

```bash
locust -f tests/performancetests/locustfile.py
```

Then navigate to `http://localhost:8089` in your browser. You'll see a web interface where you can:
- Set the number of users to simulate
- Set the spawn rate (users/second)
- Define which host to test (e.g., `http://localhost:8000`)

**Option B: Headless Mode (Recommended for CI/CD)**

```bash
# Test with 10 users, spawning 1 per second, for 1 minute
locust -f tests/performancetests/locustfile.py \
    --headless \
    --users 10 \
    --spawn-rate 1 \
    --run-time 1m \
    --host http://localhost:8000
```

**Option C: Using Environment Variable**

Set the endpoint as an environment variable:

```bash
# Linux/Mac
export MYENDPOINT=http://localhost:8000

# Windows
set MYENDPOINT=http://localhost:8000

# Then run locust
locust -f tests/performancetests/locustfile.py \
    --headless \
    --users 10 \
    --spawn-rate 1 \
    --run-time 1m \
    --host $MYENDPOINT
```

### Testing a Deployed API

If you've deployed your API to Google Cloud Run or another cloud service:

```bash
# Get the deployment URL from GCP (Linux/Mac)
export MYENDPOINT=$(gcloud run services describe <service-name> \
    --region=<region> \
    --format="value(status.url)")

# Windows
for /f "delims=" %i in ('gcloud run services describe <service-name> --region=<region> --format="value(status.url)"') do set MYENDPOINT=%i

# Run the load test
locust -f tests/performancetests/locustfile.py \
    --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --host $MYENDPOINT \
    --csv=results/locust
```

This will save results as CSV files in a `results/` directory.

## Understanding the Test Scenarios

### User Classes

1. **`MNISTAPIUser`**: Normal user behavior
   - Simulates typical API usage patterns
   - Tests root, health, and prediction endpoints
   - Realistic wait times (1-3 seconds between requests)
   - Weight distribution:
     - 10x: Prediction requests (main functionality)
     - 2x: Health checks (monitoring)
     - 1x: Root endpoint, edge cases

2. **`StressTestUser`**: Stress testing
   - Aggressive request patterns
   - Minimal wait time (0.1-0.5 seconds)
   - Helps identify performance bottlenecks

### Key Metrics to Monitor

Following DTU MLOps course material, pay attention to:

1. **Average Response Time**: How fast the API responds on average
2. **99th Percentile Response Time**: Worst-case performance (important for user experience)
3. **Requests per Second (RPS)**: API throughput
4. **Failure Rate**: Percentage of failed requests
5. **Peak Capacity**: Maximum concurrent users before degradation

### Example Output

```
Name                          # reqs      # fails  |     Avg     Min     Max  Median  |   req/s failures/s
----------------------------------------------------------------------------------------
GET /                            100     0(0.00%)  |      15       5      45      12  |    5.00    0.00
GET /health                      200     0(0.00%)  |      12       4      38      10  |   10.00    0.00
POST /predict                   1000     0(0.00%)  |     125      85     450     120  |   50.00    0.00
----------------------------------------------------------------------------------------
Aggregated                      1300     0(0.00%)  |      95       4     450      85  |   65.00    0.00
```

## Model Performance Tests

The `test_model.py` file contains performance tests that ensure:
- Model inference stays within acceptable time limits
- Predictions complete within the SLA (Service Level Agreement)

Run these tests with:

```bash
pytest tests/performancetests/test_model.py -v
```

## Integration with CI/CD

Add the following to your GitHub Actions workflow (`.github/workflows/`):

```yaml
- name: Run load tests
  env:
    DEPLOYED_MODEL_URL: ${{ env.DEPLOYED_MODEL_URL }}
  run: |
    locust -f tests/performancetests/locustfile.py \
      --headless \
      --users 100 \
      --spawn-rate 10 \
      --run-time 2m \
      --host=$DEPLOYED_MODEL_URL \
      --csv=results/locust

- name: Upload test results
  uses: actions/upload-artifact@v4
  with:
    name: locust-results
    path: results/
```

## Troubleshooting

### API Not Found (Connection Refused)
- Ensure the API server is running
- Check that you're using the correct host and port
- Verify firewall settings if testing remotely

### High Failure Rate
- Check API logs for errors
- Reduce the number of users or spawn rate
- Verify the API can handle the load (may need scaling)

### Timeout Errors
- Increase request timeout in Locust
- Check if the API/model is slow (optimize if needed)
- Consider horizontal scaling for production

## Course Alignment (DTU MLOps)

This implementation follows:
- **M24 Module**: API Testing & Load Testing
- Uses **Locust** framework as recommended
- Implements both **web UI** and **headless** modes
- Tests both **functionality** (integration tests) and **performance** (load tests)
- Follows course structure for test organization (`tests/performancetests/`)

## References

- [DTU MLOps Course - Testing APIs](https://skaftenicki.github.io/dtu_mlops/s7_deployment/testing_apis/)
- [Locust Documentation](https://docs.locust.io/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
