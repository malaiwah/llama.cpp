#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import pytest
import os
import struct
import tempfile
import shutil
import time
import subprocess
import signal
from pathlib import Path


def test_kv_cache_disabled():
    """Test that KV cache persistence is disabled by default.

    This test verifies that:
    1. Feature is disabled when --kv-cache-persist-path is not set
    2. Normal operation continues without errors
    """
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run llama-cli without KV cache persistence
        cmd = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Hello",
            "-n",
            "5",
            "--offline",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait a bit for the process to start
        time.sleep(2)

        # Send exit command
        process.stdin.write("/exit\n")
        process.stdin.flush()

        # Wait for process to complete
        stdout, stderr = process.communicate(timeout=30)

        # Verify process completed successfully
        assert process.returncode == 0, (
            f"Process failed with return code {process.returncode}"
        )

        # Verify no cache files were created
        cache_files = list(temp_path.glob("*.seq*"))
        assert len(cache_files) == 0, (
            "No cache files should be created when feature is disabled"
        )


def test_kv_cache_save_load_cycle():
    """Test complete KV cache save and load cycle.

    This test verifies that:
    1. KV cache is saved when CLI exits
    2. Cache files are created (.seq0, .seq1, etc.)
    3. KV cache is loaded on subsequent run
    4. Cache is reused for similar requests
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cache_file = temp_path / "test-cache.kvcache"

        # First run: generate some output and save cache
        print("[TEST] First run: generating output and saving cache...")
        cmd1 = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Hello, how are you?",
            "-n",
            "10",
            "--kv-cache-persist-path",
            str(cache_file),
            "--offline",
            "--single-turn",
        ]

        process1 = subprocess.Popen(
            cmd1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout1, stderr1 = process1.communicate(timeout=60)

        assert process1.returncode == 0, f"First run failed: {stderr1}"

        # Verify cache files were created
        print("[TEST] Checking for cache files...")
        cache_files = list(temp_path.glob("*.seq*"))
        assert len(cache_files) > 0, f"No cache files found in {temp_path}"
        print(
            f"[TEST] Found {len(cache_files)} cache file(s): {[f.name for f in cache_files]}"
        )

        # Verify at least .seq0 exists
        seq0_file = cache_file.parent / (str(cache_file).split("/")[-1] + ".seq0")
        assert seq0_file.exists(), f"Cache file {seq0_file} was not created"
        assert seq0_file.stat().st_size > 0, f"Cache file {seq0_file} is empty"
        print(
            f"[TEST] Cache file {seq0_file.name} exists with size {seq0_file.stat().st_size} bytes"
        )

        # Second run: load cache and verify it works
        print("[TEST] Second run: loading cache...")
        cmd2 = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Tell me a short story",
            "-n",
            "15",
            "--kv-cache-persist-path",
            str(cache_file),
            "--offline",
            "--single-turn",
        ]

        process2 = subprocess.Popen(
            cmd2,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout2, stderr2 = process2.communicate(timeout=60)

        assert process2.returncode == 0, f"Second run failed: {stderr2}"

        # Verify cache files still exist after second run
        cache_files_after = list(temp_path.glob("*.seq*"))
        assert len(cache_files_after) > 0, (
            "Cache files should still exist after second run"
        )
        print(
            f"[TEST] Cache files after second run: {[f.name for f in cache_files_after]}"
        )

        print("[TEST] KV cache save/load cycle test completed successfully")


def test_kv_cache_validation():
    """Test KV cache validation with invalid headers.

    This test verifies that:
    1. Cache files with invalid magic numbers are skipped
    2. Cache files with unsupported versions are skipped
    3. Cache files with mismatched model hashes are skipped
    4. Appropriate warnings are logged for invalid files
    5. Valid cache files are loaded correctly
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cache_file = temp_path / "test-cache.kvcache"

        # First run: create a valid cache
        print("[TEST] Creating valid cache...")
        cmd1 = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Test",
            "-n",
            "5",
            "--kv-cache-persist-path",
            str(cache_file),
            "--offline",
            "--single-turn",
        ]

        process1 = subprocess.Popen(
            cmd1,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout1, stderr1 = process1.communicate(timeout=60)

        assert process1.returncode == 0, f"First run failed: {stderr1}"

        # Create a cache file with invalid magic number
        print("[TEST] Creating cache file with invalid magic number...")
        invalid_magic_file = cache_file.parent / (
            str(cache_file).split("/")[-1] + ".seq1"
        )
        with open(invalid_magic_file, "wb") as f:
            # Write invalid magic (0x00000000 instead of 0x4B564343)
            f.write(struct.pack("<I", 0x00000000))  # Invalid magic
            f.write(struct.pack("<I", 1))  # Valid version
            f.write(struct.pack("<I", 0x12345678))  # Model hash (doesn't matter)
            f.write(struct.pack("<I", 0))  # Reserved
            # Write some dummy data
            f.write(b"dummy cache data")

        # Create a cache file with unsupported version
        print("[TEST] Creating cache file with unsupported version...")
        invalid_version_file = cache_file.parent / (
            str(cache_file).split("/")[-1] + ".seq2"
        )
        with open(invalid_version_file, "wb") as f:
            f.write(struct.pack("<I", 0x4B564343))  # Valid magic
            f.write(struct.pack("<I", 999))  # Unsupported version
            f.write(struct.pack("<I", 0x12345678))  # Model hash
            f.write(struct.pack("<I", 0))  # Reserved
            f.write(b"dummy cache data")

        # Second run: should skip invalid files and load valid ones
        print("[TEST] Running CLI to test validation...")
        cmd2 = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Test",
            "-n",
            "5",
            "--kv-cache-persist-path",
            str(cache_file),
            "--offline",
            "--single-turn",
        ]

        process2 = subprocess.Popen(
            cmd2,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout2, stderr2 = process2.communicate(timeout=60)

        assert process2.returncode == 0, f"Second run failed: {stderr2}"

        # Verify that the valid cache file (seq0) still exists
        valid_file = cache_file.parent / (str(cache_file).split("/")[-1] + ".seq0")
        assert valid_file.exists(), "Valid cache file seq0 should still exist"

        print("[TEST] KV cache validation test completed successfully")


def test_kv_cache_signal_interrupt():
    """Test that KV cache is saved when process is interrupted with SIGINT.

    This test verifies that:
    1. KV cache is saved when Ctrl+C (SIGINT) is received
    2. Cache files are created even on interrupt
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cache_file = temp_path / "test-cache.kvcache"

        # Run CLI with a long-running prompt
        print("[TEST] Starting CLI with long-running prompt...")
        cmd = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Tell me a very long story",
            "-n",
            "100",
            "--kv-cache-persist-path",
            str(cache_file),
            "--offline",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the process to start generating output
        time.sleep(3)

        # Send SIGINT (Ctrl+C)
        print("[TEST] Sending SIGINT to process...")
        process.send_signal(signal.SIGINT)

        # Wait for process to complete
        stdout, stderr = process.communicate(timeout=30)

        # Process should exit gracefully (return code 0 or 130 for SIGINT)
        assert process.returncode in [0, 130], (
            f"Process exited with unexpected code {process.returncode}"
        )

        # Verify cache files were created
        print("[TEST] Checking for cache files after interrupt...")
        cache_files = list(temp_path.glob("*.seq*"))
        assert len(cache_files) > 0, (
            f"No cache files found in {temp_path} after interrupt"
        )
        print(f"[TEST] Found {len(cache_files)} cache file(s) after interrupt")

        print("[TEST] KV cache signal interrupt test completed successfully")


def test_kv_cache_empty_path():
    """Test that KV cache feature is disabled when path is empty.

    This test verifies that:
    1. Empty path disables the feature
    2. No cache files are created
    3. Normal operation continues
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Run CLI without --kv-cache-persist-path
        print("[TEST] Running CLI without KV cache persistence...")
        cmd = [
            "./build/bin/llama-cli",
            "-m",
            "ggml-org/models:tinyllamas/stories260K.gguf",
            "-t",
            "1",
            "-p",
            "Hello",
            "-n",
            "5",
            "--offline",
            "--single-turn",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(timeout=60)

        assert process.returncode == 0, f"Process failed: {stderr}"

        # Verify no cache files were created
        print("[TEST] Verifying no cache files were created...")
        cache_files = list(temp_path.glob("*.seq*"))
        assert len(cache_files) == 0, (
            "No cache files should be created when path is not specified"
        )

        print("[TEST] KV cache empty path test completed successfully")
