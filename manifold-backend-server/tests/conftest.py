"""Test-Konfiguration und Fixtures für pytest"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil

# Import der App
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app, UPLOAD_DIR


@pytest.fixture
def client():
    """Erstellt einen TestClient für die FastAPI-App"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_upload_dir(tmp_path):
    """Erstellt ein temporäres Upload-Verzeichnis für Tests"""
    test_dir = tmp_path / "test_uploads"
    test_dir.mkdir()
    
    # Backup des originalen UPLOAD_DIR
    original_dir = UPLOAD_DIR
    
    yield test_dir
    
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_file(tmp_path):
    """Erstellt eine Test-Datei"""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Dies ist eine Test-Datei für den Upload")
    return file_path


@pytest.fixture
def sample_csv(tmp_path):
    """Erstellt eine Test-CSV-Datei"""
    file_path = tmp_path / "test_data.csv"
    file_path.write_text("col1,col2,col3\n1,2,3\n4,5,6")
    return file_path

