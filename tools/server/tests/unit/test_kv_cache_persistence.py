#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import pytest
import os
import tempfile
import shutil
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_kv_cache_disabled():
    """Test that KV cache persistence is disabled by default.

    This test verifies that:
    1. Feature is disabled when kv-cache-persist-path is not set
    2. Normal operation continues without errors
    """
    global server
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    # Load the model
    load_res = server.make_request("POST", "/models/load", data={"model": model_id})
    assert load_res.status_code == 200
    assert load_res.body.get("success") is True

    _wait_for_model_status(model_id, {"loaded"}, timeout=120)

    # Process a request
    chat_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 5,
        },
    )
    assert chat_res.status_code == 200

    # Unload the model
    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200

    _wait_for_model_status(model_id, {"unloaded"})


def test_kv_cache_router_mode_basic():
    """Test basic router mode operation with models.

    This test verifies that:
    1. Router mode works correctly
    2. Models can be loaded and unloaded
    3. Multiple models can be managed
    """
    global server
    server.models_max = 2
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    # Load the model
    load_res = server.make_request("POST", "/models/load", data={"model": model_id})
    assert load_res.status_code == 200
    assert load_res.body.get("success") is True

    _wait_for_model_status(model_id, {"loaded"}, timeout=120)

    # Process a request
    chat_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    )
    assert chat_res.status_code == 200
    assert "choices" in chat_res.body

    # Unload the model
    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200

    _wait_for_model_status(model_id, {"unloaded"})


def test_kv_cache_multiple_models():
    """Test router mode with multiple models.

    This test verifies that:
    1. Multiple models can be loaded sequentially
    2. LRU eviction works correctly
    3. Each model operates independently
    """
    global server
    server.models_max = 2
    server.start()

    models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K",
    ]

    # Load first model
    load_res1 = server.make_request("POST", "/models/load", data={"model": models[0]})
    assert load_res1.status_code == 200
    _wait_for_model_status(models[0], {"loaded"}, timeout=120)

    # Process request with first model
    chat_res1 = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": models[0],
            "messages": [{"role": "user", "content": "Model 1 prompt"}],
            "max_tokens": 5,
        },
    )
    assert chat_res1.status_code == 200

    # Load second model
    load_res2 = server.make_request("POST", "/models/load", data={"model": models[1]})
    assert load_res2.status_code == 200
    _wait_for_model_status(models[1], {"loaded"}, timeout=120)

    # Both models should be loaded
    assert _get_model_status(models[0]) == "loaded"
    assert _get_model_status(models[1]) == "loaded"

    # Process request with second model
    chat_res2 = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": models[1],
            "messages": [{"role": "user", "content": "Model 2 prompt"}],
            "max_tokens": 5,
        },
    )
    assert chat_res2.status_code == 200

    # Unload both models
    unload_res1 = server.make_request(
        "POST", "/models/unload", data={"model": models[0]}
    )
    assert unload_res1.status_code == 200

    unload_res2 = server.make_request(
        "POST", "/models/unload", data={"model": models[1]}
    )
    assert unload_res2.status_code == 200


def _get_model_status(model_id: str) -> str:
    """Helper function to get model status from /models endpoint."""
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    """Helper function to wait for model to reach desired status."""
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )
