"""Tests für die Haupt-API-Endpoints"""

import pytest
from fastapi.testclient import TestClient


def test_read_root(client):
    """Test des Root-Endpoints"""
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json() == {"message": "Willkommen zur Manifold API"}


def test_health_check(client):
    """Test des Health-Check-Endpoints"""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_returns_json(client):
    """Überprüft, dass Root JSON zurückgibt"""
    response = client.get("/")
    
    assert response.headers["content-type"] == "application/json"


def test_health_returns_json(client):
    """Überprüft, dass Health JSON zurückgibt"""
    response = client.get("/health")
    
    assert response.headers["content-type"] == "application/json"


def test_nonexistent_endpoint(client):
    """Test für nicht existierenden Endpoint"""
    response = client.get("/nonexistent")
    
    assert response.status_code == 404


def test_cors_headers(client):
    """Test CORS-Headers"""
    response = client.get("/", headers={"Origin": "http://localhost:3000"})
    
    # CORS-Header sollten gesetzt sein
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"

