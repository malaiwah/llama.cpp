import pytest
import time
import tempfile
import os
from utils import *

server: ServerProcess


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


def test_hot_reload_add_new_model():
    """Test that adding a new model to the preset file is detected and loaded."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-1]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 2
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-1" in model_ids

        time.sleep(3)

        with open(config_file, "a") as f:
            f.write("\n[test-model-2]\n")
            f.write("hf_repo = ggml-org/test-model-stories260K-infill\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-1" in model_ids
        assert "test-model-2" in model_ids
    finally:
        os.unlink(config_file)


def test_hot_reload_update_existing_model():
    """Test that updating an existing model config is detected."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-update]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")
        f.write("ctx_size = 512\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 2
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model = next((m for m in models if m.get("id") == "test-model-update"), None)
        assert model is not None

        time.sleep(3)

        with open(config_file, "w") as f:
            f.write("version = 1\n")
            f.write("\n[test-model-update]\n")
            f.write("hf_repo = ggml-org/test-model-stories260K\n")
            f.write("ctx_size = 1024\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model = next((m for m in models if m.get("id") == "test-model-update"), None)
        assert model is not None
    finally:
        os.unlink(config_file)


def test_hot_reload_remove_model():
    """Test that removing a model from the preset file is detected."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-remove-1]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")
        f.write("\n[test-model-remove-2]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K-infill\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 2
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-remove-1" in model_ids
        assert "test-model-remove-2" in model_ids

        time.sleep(3)

        with open(config_file, "w") as f:
            f.write("version = 1\n")
            f.write("\n[test-model-remove-1]\n")
            f.write("hf_repo = ggml-org/test-model-stories260K\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model_ids = [m.get("id") for m in models]
        assert "test-model-remove-1" in model_ids

        removed_model = next(
            (m for m in models if m.get("id") == "test-model-remove-2"), None
        )
        assert removed_model is not None
        assert removed_model.get("status", {}).get("value") == "unloaded"
    finally:
        os.unlink(config_file)


def test_hot_reload_invalid_config():
    """Test that invalid config does not break the server and previous config remains active."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-invalid]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 2
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-invalid" in model_ids

        time.sleep(3)

        with open(config_file, "w") as f:
            f.write("version = 1\n")
            f.write("\n[broken-model]\n")
            f.write("hf_repo = non-existent-repo\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model_ids = [m.get("id") for m in models]

        assert "test-model-invalid" in model_ids
        assert "broken-model" in model_ids

        broken_model = next((m for m in models if m.get("id") == "broken-model"), None)
        assert broken_model is not None
        assert broken_model.get("status", {}).get("value") == "unloaded"

        valid_model = next(
            (m for m in models if m.get("id") == "test-model-invalid"), None
        )
        assert valid_model is not None
        assert valid_model.get("status", {}).get("value") == "unloaded"
    finally:
        os.unlink(config_file)


def test_hot_reload_custom_interval():
    """Test that custom watch interval is respected."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-interval]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 1
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-interval" in model_ids

        time.sleep(2)

        with open(config_file, "a") as f:
            f.write("\n[test-model-interval-2]\n")
            f.write("hf_repo = ggml-org/test-model-stories260K-infill\n")

        time.sleep(3)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-interval" in model_ids
        assert "test-model-interval-2" in model_ids
    finally:
        os.unlink(config_file)


def test_hot_reload_multiple_changes():
    """Test that multiple rapid changes to config file are handled correctly."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-multi]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 1
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-multi" in model_ids

        time.sleep(2)

        for i in range(2, 4):
            with open(config_file, "a") as f:
                f.write(f"\n[test-model-multi-{i}]\n")
                f.write("hf_repo = ggml-org/test-model-stories260K-infill\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        model_ids = [m.get("id") for m in res.body.get("data", [])]
        assert "test-model-multi" in model_ids
        assert "test-model-multi-2" in model_ids
        assert "test-model-multi-3" in model_ids
    finally:
        os.unlink(config_file)


def test_hot_reload_with_loaded_model():
    """Test that hot reload works correctly when models are already loaded."""
    global server

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
        config_file = f.name
        f.write("version = 1\n")
        f.write("\n[test-model-loaded]\n")
        f.write("hf_repo = ggml-org/test-model-stories260K\n")

    try:
        server.models_preset = config_file
        server.models_preset_watch = True
        server.models_preset_watch_interval = 2
        server.no_models_autoload = True
        server.start()

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model = next((m for m in models if m.get("id") == "test-model-loaded"), None)
        assert model is not None
        assert model.get("status", {}).get("value") == "unloaded"

        time.sleep(3)

        with open(config_file, "w") as f:
            f.write("version = 1\n")
            f.write("\n[test-model-loaded]\n")
            f.write("hf_repo = ggml-org/test-model-stories260K\n")
            f.write("temperature = 0.9\n")

        time.sleep(5)

        res = server.make_request("GET", "/models")
        assert res.status_code == 200
        models = res.body.get("data", [])
        model = next((m for m in models if m.get("id") == "test-model-loaded"), None)
        assert model is not None

        assert model.get("status", {}).get("value") == "unloaded"
    finally:
        os.unlink(config_file)
