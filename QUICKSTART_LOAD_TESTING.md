# Quick Start Guide: API Load Testing

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
# Using uv (recommended)
uv sync --group dev

# Or manually install
pip install httpx locust fastapi uvicorn
```

### Step 2: Start the API
```bash
# Terminal 1: Start the API server
uvicorn uaag2.api:app --reload
```

### Step 3: Run Tests
```bash
# Terminal 2: Run integration tests
pytest tests/test_api.py -v

# Terminal 3: Run load tests (web UI)
locust -f tests/performancetests/locustfile.py
# Open http://localhost:8089 and enter:
# - Host: http://localhost:8000
# - Users: 10
# - Spawn rate: 1
```

## 📊 Quick Commands Reference

### API Commands
```bash
# Start API
uvicorn uaag2.api:app --reload

# Start API with uv
uv run uvicorn uaag2.api:app --reload

# Test API manually
curl http://localhost:8000/
curl http://localhost:8000/health

# Interactive API docs
# Open: http://localhost:8000/docs
```

### Test Commands
```bash
# Run all API tests
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::TestAPIEndpoints::test_health_check -v

# Run with coverage
pytest tests/test_api.py --cov=uaag2.api
```

### Load Test Commands
```bash
# Web UI mode (interactive)
locust -f tests/performancetests/locustfile.py

# Headless mode (10 users, 1 min)
locust -f tests/performancetests/locustfile.py \
    --headless -u 10 -r 1 -t 1m \
    --host http://localhost:8000

# Headless mode with results export
locust -f tests/performancetests/locustfile.py \
    --headless -u 100 -r 10 -t 5m \
    --host http://localhost:8000 \
    --csv=results/locust \
    --html=results/report.html
```

## 🎯 What to Check

### Integration Tests Should Show:
```
tests/test_api.py::TestAPIEndpoints::test_read_root PASSED
tests/test_api.py::TestAPIEndpoints::test_health_check PASSED
tests/test_api.py::TestAPIEndpoints::test_predict_valid_input PASSED
... (11 tests total)
```

### Load Tests Should Show:
```
Name                          # reqs  # fails  Avg    Min    Max  Median  req/s
-----------------------------------------------------------------------------
GET /                            100      0     15      5     45      12    5.0
GET /health                      200      0     12      4     38      10   10.0
POST /predict                   1000      0    125     85    450     120   50.0
-----------------------------------------------------------------------------
Aggregated                      1300      0     95      4    450      85   65.0
```

## 🐛 Troubleshooting

### Problem: "Connection refused"
**Solution**: Make sure the API is running
```bash
uvicorn uaag2.api:app --reload
```

### Problem: "Module not found: uaag2"
**Solution**: Install the package
```bash
uv sync
# or
pip install -e .
```

### Problem: "locust: command not found"
**Solution**: Install locust
```bash
uv sync --group dev
# or
pip install locust
```

### Problem: "Model file not found"
**Note**: This is OK for testing! The API will use an untrained model.
To use a trained model:
```bash
# Set the model path
export MODEL_PATH=/path/to/your/model.pth
```

## 📁 Project Structure
```
UAAG2/
├── src/uaag2/
│   └── api.py                    # FastAPI application
├── tests/
│   ├── test_api.py               # Integration tests
│   └── performancetests/
│       ├── locustfile.py         # Load tests
│       └── README.md             # Detailed guide
├── reports/
│   ├── API_TESTING.md            # Full documentation
│   └── LOAD_TESTING_SUMMARY.md   # Implementation summary
└── scripts/
    └── api_example.py            # Usage example
```

## 🎓 DTU MLOps Checklist (M24)

- [x] httpx installed
- [x] locust installed
- [x] FastAPI application created
- [x] Integration tests created
- [x] Load tests created
- [x] Tests in correct directory structure
- [x] Documentation provided
- [x] Ready for CI/CD

## 🔗 Useful Links

- API Docs: http://localhost:8000/docs (when running)
- Locust UI: http://localhost:8089 (when running)
- [Full Load Testing Guide](tests/performancetests/README.md)
- [API Testing Documentation](reports/API_TESTING.md)
- [Implementation Summary](reports/LOAD_TESTING_SUMMARY.md)
- [DTU MLOps Course Material](https://skaftenicki.github.io/dtu_mlops/s7_deployment/testing_apis/)

## 💡 Tips

1. **Always start the API first** before running tests
2. **Use web UI mode** for exploration and debugging
3. **Use headless mode** for CI/CD and automated testing
4. **Check the logs** if something doesn't work
5. **Start small** (10 users) and increase gradually

---
**Quick Help**: For more details, see `tests/performancetests/README.md` or `reports/API_TESTING.md`
