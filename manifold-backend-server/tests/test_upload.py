"""Tests für den Upload-Endpoint"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO


def test_upload_file_success(client, sample_file):
    """Test: Erfolgreicher Datei-Upload"""
    with open(sample_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test_file.txt", f, "text/plain")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_file.txt"
    assert "size" in data
    assert data["size"] > 0


def test_upload_csv_file(client, sample_csv):
    """Test: CSV-Datei hochladen"""
    with open(sample_csv, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test_data.csv", f, "text/csv")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_data.csv"
    assert data["content_type"] == "text/csv"


def test_upload_without_file(client):
    """Test: Upload ohne Datei sollte fehlschlagen"""
    response = client.post("/upload")
    
    assert response.status_code == 422  # Unprocessable Entity


def test_upload_empty_file(client):
    """Test: Leere Datei hochladen"""
    file_content = BytesIO(b"")
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", file_content, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["size"] == 0


def test_upload_large_filename(client):
    """Test: Datei mit langem Namen"""
    long_filename = "a" * 200 + ".txt"
    file_content = BytesIO(b"test content")
    
    response = client.post(
        "/upload",
        files={"file": (long_filename, file_content, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == long_filename


def test_upload_binary_file(client, tmp_path):
    """Test: Binärdatei hochladen"""
    # Erstelle eine kleine "binäre" Datei
    binary_file = tmp_path / "test.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    
    with open(binary_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test.bin", f, "application/octet-stream")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["size"] == 6


def test_upload_response_structure(client, sample_file):
    """Test: Überprüft die Struktur der Upload-Response"""
    with open(sample_file, "rb") as f:
        response = client.post(
            "/upload",
            files={"file": ("test_file.txt", f, "text/plain")}
        )
    
    data = response.json()
    
    # Überprüfe, dass alle erwarteten Felder vorhanden sind
    assert "filename" in data
    assert "status" in data
    assert "size" in data
    assert "content_type" in data
    assert "path" in data


def test_upload_multiple_files_sequential(client, sample_file, sample_csv):
    """Test: Mehrere Dateien nacheinander hochladen"""
    # Erste Datei
    with open(sample_file, "rb") as f:
        response1 = client.post(
            "/upload",
            files={"file": ("file1.txt", f, "text/plain")}
        )
    assert response1.status_code == 200
    
    # Zweite Datei
    with open(sample_csv, "rb") as f:
        response2 = client.post(
            "/upload",
            files={"file": ("file2.csv", f, "text/csv")}
        )
    assert response2.status_code == 200
    
    # Beide sollten erfolgreich sein
    assert response1.json()["status"] == "success"
    assert response2.json()["status"] == "success"


def test_upload_special_characters_in_filename(client):
    """Test: Dateiname mit Sonderzeichen"""
    filename = "test_äöü_file.txt"
    file_content = BytesIO(b"test content")
    
    response = client.post(
        "/upload",
        files={"file": (filename, file_content, "text/plain")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

