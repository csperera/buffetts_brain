# Test Suite for Buffett's Brain

## Running Tests
```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag_pipeline.py

# Run with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_query_routing.py::TestTimeSensitiveDetection::test_time_sensitive_detection
```

## Test Structure

- `test_rag_pipeline.py`: RAG retrieval and document processing
- `test_query_routing.py`: Intelligent routing logic
- `test_embeddings.py`: Embedding generation and operations
- `test_integration.py`: End-to-end integration tests

## Test Coverage

Current test coverage focuses on:
- ✅ Query routing logic
- ✅ Time-sensitive detection
- ✅ Relevance evaluation
- ✅ Embedding operations
- ✅ Configuration validation
- ✅ Error handling

## Notes

Some tests require valid API keys to run. Tests will skip automatically if keys are not available.

For CI/CD integration, consider:
- Mocking all external API calls
- Using test fixtures for vector DB
- Running tests in isolation