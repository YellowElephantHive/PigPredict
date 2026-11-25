import os
import json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PigPredict.settings")
import django
django.setup()
from django.test import Client, override_settings
from PigPredict_app.runtime import predict_recipe
import pytest


@pytest.fixture
def client():
    with override_settings(ALLOWED_HOSTS=["testserver"]):
        yield Client(enforce_csrf_checks=True)


def test_studio_and_legacy_links(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'<html lang="en">' in response.content
    assert b'language-zh' in response.content
    for path in ("/forms/", "/training/"):
        assert client.get(path).status_code == 302
    assert client.get("/api/health/").json()["parameters"] == 109923


def test_api_preserves_csrf_and_returns_real_inference(client):
    payload = {"fabric":"FVF2184", "dyes":["228-9","261-8","327"], "concentrations":[.2,.4,.6]}
    assert client.post("/api/predict/", data=json.dumps(payload), content_type="application/json").status_code == 403
    client.get("/")
    response = client.post("/api/predict/", data=json.dumps(payload), content_type="application/json",
                           HTTP_X_CSRFTOKEN=client.cookies["csrftoken"].value)
    assert response.status_code == 200
    assert response.json() == predict_recipe(payload)


@pytest.mark.parametrize("payload", [None, [], {}, {"fabric": "unknown"},
    {"fabric":"FVF2184", "dyes":["122"], "concentrations":[-1]},
    {"fabric":"FVF2184", "dyes":["122"], "concentrations":[float("nan")]},
    {"fabric":"FVF2184", "dyes":["122"], "concentrations":[True]},
    {"fabric":"FVF2184", "dyes":["122","122"], "concentrations":[1,2]},
    {"fabric":"FVF2184", "dyes":["not-a-dye"], "concentrations":[1]},
])
def test_api_rejects_invalid_inputs(client, payload):
    client.get("/")
    response = client.post("/api/predict/", data=json.dumps(payload), content_type="application/json",
                           HTTP_X_CSRFTOKEN=client.cookies["csrftoken"].value)
    assert response.status_code == 400
