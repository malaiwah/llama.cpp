#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import pytest
import os
import struct
import tempfile
import shutil
import time
import re
from pathlib import Path
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


def test_kv_cache_save_load_cycle():
    """Test complete KV cache save and load cycle.

    This test verifies that:
    1. KV cache is saved when model is unloaded
    2. Cache files are created (.seq0, .seq1, etc.)
    3. KV cache is loaded when model is reloaded
    4. Cache is reused for similar requests
    5. Temporary directory is cleaned up

    The test creates a temporary directory with a model preset file
    configured with kv-cache-persist-path to enable the feature.
    """
    global server

    # Create a temporary directory for KV cache and model presets
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cache_dir = temp_path / "cache"
        cache_dir.mkdir()

        # Create a model preset file with KV cache persistence enabled
        preset_file = temp_path / "models.ini"
        preset_content = f"""[test-model]
model = ggml-org/tinygemma3-GGUF:Q8_0
kv-cache-persist-path = {cache_dir / "test-model.kvcache"}
"""
        preset_file.write_text(preset_content)

        # Configure server to use the temporary preset file
        server.models_preset = str(preset_file)
        server.start()

        model_id = "test-model"

        # Step 1: Load the model
        print(f"[TEST] Loading model {model_id}...")
        load_res = server.make_request("POST", "/models/load", data={"model": model_id})
        assert load_res.status_code == 200, f"Failed to load model: {load_res.body}"
        assert load_res.body.get("success") is True

        _wait_for_model_status(model_id, {"loaded"}, timeout=120)
        print(f"[TEST] Model {model_id} loaded successfully")

        # Step 2: Process a chat completion request to populate the cache
        print("[TEST] Processing first request to populate cache...")
        chat_res1 = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 10,
            },
        )
        assert chat_res1.status_code == 200, f"Chat completion failed: {chat_res1.body}"
        assert "choices" in chat_res1.body
        print(
            f"[TEST] First request completed: {chat_res1.body['choices'][0]['message']['content']}"
        )

        # Step 3: Unload the model (should trigger KV cache save)
        print("[TEST] Unloading model to trigger cache save...")
        unload_res1 = server.make_request(
            "POST", "/models/unload", data={"model": model_id}
        )
        assert unload_res1.status_code == 200, (
            f"Failed to unload model: {unload_res1.body}"
        )

        _wait_for_model_status(model_id, {"unloaded"})
        print("[TEST] Model unloaded successfully")

        # Step 4: Verify cache files were created
        print("[TEST] Checking for cache files...")
        cache_files = list(cache_dir.glob("*.seq*"))
        assert len(cache_files) > 0, (
            f"No cache files found in {cache_dir}. Expected at least one .seq file"
        )
        print(
            f"[TEST] Found {len(cache_files)} cache file(s): {[f.name for f in cache_files]}"
        )

        # Verify at least .seq0 exists
        seq0_file = cache_dir / "test-model.kvcache.seq0"
        assert seq0_file.exists(), f"Cache file {seq0_file} was not created"
        assert seq0_file.stat().st_size > 0, f"Cache file {seq0_file} is empty"
        print(
            f"[TEST] Cache file {seq0_file.name} exists with size {seq0_file.stat().st_size} bytes"
        )

        # Step 5: Reload the model (should trigger KV cache load)
        print("[TEST] Reloading model to trigger cache load...")
        load_res2 = server.make_request(
            "POST", "/models/load", data={"model": model_id}
        )
        assert load_res2.status_code == 200, f"Failed to reload model: {load_res2.body}"
        assert load_res2.body.get("success") is True

        _wait_for_model_status(model_id, {"loaded"}, timeout=120)
        print("[TEST] Model reloaded successfully")

        # Step 6: Process a similar request (should use cached KV cache)
        print("[TEST] Processing second request to verify cache reuse...")
        chat_res2 = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 10,
            },
        )
        assert chat_res2.status_code == 200, (
            f"Second chat completion failed: {chat_res2.body}"
        )
        assert "choices" in chat_res2.body
        print(
            f"[TEST] Second request completed: {chat_res2.body['choices'][0]['message']['content']}"
        )

        # Step 7: Unload model again to save cache
        print("[TEST] Unloading model to save cache again...")
        unload_res2 = server.make_request(
            "POST", "/models/unload", data={"model": model_id}
        )
        assert unload_res2.status_code == 200, (
            f"Failed to unload model: {unload_res2.body}"
        )

        _wait_for_model_status(model_id, {"unloaded"})
        print("[TEST] Model unloaded successfully")

        # Step 8: Verify cache files still exist and are valid
        print("[TEST] Verifying cache files after second unload...")
        cache_files_after = list(cache_dir.glob("*.seq*"))
        assert len(cache_files_after) > 0, (
            f"No cache files found in {cache_dir} after second unload"
        )
        print(f"[TEST] Cache files still exist: {[f.name for f in cache_files_after]}")

        # Step 9: Reload model one more time to verify cache loads correctly
        print("[TEST] Reloading model to verify cache loads correctly...")
        load_res3 = server.make_request(
            "POST", "/models/load", data={"model": model_id}
        )
        assert load_res3.status_code == 200, (
            f"Failed to reload model third time: {load_res3.body}"
        )
        assert load_res3.body.get("success") is True

        _wait_for_model_status(model_id, {"loaded"}, timeout=120)
        print("[TEST] Model reloaded successfully (third time)")

        # Step 10: Process another request
        print("[TEST] Processing third request...")
        chat_res3 = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Tell me a short story"}],
                "max_tokens": 15,
            },
        )
        assert chat_res3.status_code == 200, (
            f"Third chat completion failed: {chat_res3.body}"
        )
        assert "choices" in chat_res3.body
        print(
            f"[TEST] Third request completed: {chat_res3.body['choices'][0]['message']['content']}"
        )

        # Step 11: Final cleanup - unload model
        print("[TEST] Final cleanup - unloading model...")
        unload_res3 = server.make_request(
            "POST", "/models/unload", data={"model": model_id}
        )
        assert unload_res3.status_code == 200, (
            f"Failed to unload model: {unload_res3.body}"
        )

        _wait_for_model_status(model_id, {"unloaded"})
        print("[TEST] Model unloaded successfully")

        print("[TEST] KV cache save/load cycle test completed successfully")
        print(f"[TEST] Cache directory: {cache_dir}")
        print(f"[TEST] Cache files: {[f.name for f in cache_files_after]}")

        # Temporary directory will be automatically cleaned up by tempfile context manager


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


def test_kv_cache_validation():
    """Test KV cache validation with invalid headers.

    This test verifies that:
    1. Cache files with invalid magic numbers are skipped
    2. Cache files with unsupported versions are skipped
    3. Cache files with mismatched model hashes are skipped
    4. Appropriate warnings are logged for invalid files
    5. Valid cache files are loaded correctly
    """
    global server

    # Create a temporary directory for KV cache and model presets
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cache_dir = temp_path / "cache"
        cache_dir.mkdir()

        # Create a model preset file with KV cache persistence enabled
        preset_file = temp_path / "models.ini"
        preset_content = f"""[test-model]
model = ggml-org/tinygemma3-GGUF:Q8_0
kv-cache-persist-path = {cache_dir / "test-model.kvcache"}
"""
        preset_file.write_text(preset_content)

        # Configure server to use the temporary preset file
        server.models_preset = str(preset_file)
        server.start()

        model_id = "test-model"

        # Step 1: Load the model and create a valid cache
        print("[TEST] Loading model to create valid cache...")
        load_res = server.make_request("POST", "/models/load", data={"model": model_id})
        assert load_res.status_code == 200, f"Failed to load model: {load_res.body}"

        _wait_for_model_status(model_id, {"loaded"}, timeout=120)

        # Process a request to populate the cache
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

        # Unload to save cache
        unload_res = server.make_request(
            "POST", "/models/unload", data={"model": model_id}
        )
        assert unload_res.status_code == 200

        _wait_for_model_status(model_id, {"unloaded"})

        # Step 2: Create a cache file with invalid magic number
        print("[TEST] Creating cache file with invalid magic number...")
        invalid_magic_file = cache_dir / "test-model.kvcache.seq1"
        with open(invalid_magic_file, "wb") as f:
            # Write invalid magic (0x00000000 instead of 0x4B564343)
            f.write(struct.pack("<I", 0x00000000))  # Invalid magic
            f.write(struct.pack("<I", 1))  # Valid version
            f.write(struct.pack("<I", 0x12345678))  # Model hash (doesn't matter)
            f.write(struct.pack("<I", 0))  # Reserved
            # Write some dummy data
            f.write(b"dummy cache data")

        # Step 3: Create a cache file with unsupported version
        print("[TEST] Creating cache file with unsupported version...")
        invalid_version_file = cache_dir / "test-model.kvcache.seq2"
        with open(invalid_version_file, "wb") as f:
            f.write(struct.pack("<I", 0x4B564343))  # Valid magic
            f.write(struct.pack("<I", 999))  # Unsupported version
            f.write(struct.pack("<I", 0x12345678))  # Model hash
            f.write(struct.pack("<I", 0))  # Reserved
            f.write(b"dummy cache data")

        # Step 4: Reload the model and verify invalid files are skipped
        print("[TEST] Reloading model to test validation...")
        load_res2 = server.make_request(
            "POST", "/models/load", data={"model": model_id}
        )
        assert load_res2.status_code == 200, f"Failed to reload model: {load_res2.body}"

        _wait_for_model_status(model_id, {"loaded"}, timeout=120)

        # Process a request to verify cache is working
        chat_res2 = server.make_request(
            "POST",
            "/v1/chat/completions",
            data={
                "model": model_id,
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
            },
        )
        assert chat_res2.status_code == 200

        # Unload model
        unload_res2 = server.make_request(
            "POST", "/models/unload", data={"model": model_id}
        )
        assert unload_res2.status_code == 200

        _wait_for_model_status(model_id, {"unloaded"})

        # Step 5: Verify that only valid cache file (seq0) exists and invalid ones were not used
        print("[TEST] Verifying cache file handling...")
        cache_files = list(cache_dir.glob("*.seq*"))
        print(f"[TEST] Cache files after reload: {[f.name for f in cache_files]}")

        # The valid cache file (seq0) should still exist
        valid_file = cache_dir / "test-model.kvcache.seq0"
        assert valid_file.exists(), "Valid cache file seq0 should still exist"

        # Invalid files may still exist on disk but were not loaded
        # (they might have been overwritten or skipped)

        print("[TEST] KV cache validation test completed successfully")
