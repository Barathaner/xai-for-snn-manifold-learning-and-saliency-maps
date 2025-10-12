# Tests für Manifold Backend Server

## Installation der Test-Dependencies

```bash
pip install pytest pytest-asyncio httpx
```

## Tests ausführen

### Alle Tests ausführen
```bash
cd manifold-backend-server
pytest
```

### Tests mit ausführlicher Ausgabe
```bash
pytest -v
```

### Nur bestimmte Test-Dateien ausführen
```bash
pytest tests/test_main.py
pytest tests/test_upload.py
```

### Nur bestimmte Tests ausführen
```bash
pytest tests/test_main.py::test_read_root
pytest tests/test_upload.py::test_upload_file_success
```

### Tests mit Coverage-Report
```bash
pip install pytest-cov
pytest --cov=. --cov-report=html
```

## Test-Struktur

```
tests/
├── __init__.py          # Markiert Verzeichnis als Python-Paket
├── conftest.py          # Fixtures und Test-Konfiguration
├── test_main.py         # Tests für Haupt-Endpoints (/, /health)
└── test_upload.py       # Tests für Upload-Endpoint
```

## Fixtures

- `client`: TestClient für die FastAPI-App
- `sample_file`: Beispiel-Textdatei für Tests
- `sample_csv`: Beispiel-CSV-Datei für Tests
- `test_upload_dir`: Temporäres Upload-Verzeichnis

## Test-Kategorien

### Unit-Tests (test_main.py)
- ✅ Root-Endpoint (`/`)
- ✅ Health-Check-Endpoint (`/health`)
- ✅ CORS-Headers
- ✅ 404-Fehler

### Integration-Tests (test_upload.py)
- ✅ Datei-Upload
- ✅ CSV-Upload
- ✅ Leere Dateien
- ✅ Binärdateien
- ✅ Mehrere Dateien
- ✅ Sonderzeichen in Dateinamen

## Beispiele

### Eigene Test-Datei erstellen

```python
# tests/test_custom.py

def test_custom_endpoint(client):
    """Test für eigenen Endpoint"""
    response = client.get("/custom")
    assert response.status_code == 200
```

### Async-Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Test für async Funktionen"""
    result = await some_async_function()
    assert result is not None
```

### Test mit Parametrisierung

```python
@pytest.mark.parametrize("input,expected", [
    ("test.txt", "success"),
    ("data.csv", "success"),
])
def test_multiple_cases(client, input, expected):
    """Test mit mehreren Parametern"""
    # Test-Logik
    pass
```

