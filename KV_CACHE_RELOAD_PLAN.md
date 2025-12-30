# KV Cache Persistence for Router Mode - Implementation Plan

## Overview

This feature implements persistent KV cache (prompt cache) saving and reloading for llama.cpp router mode. When models are unloaded in router mode, their prompt cache will be saved to disk and automatically reloaded when the model is loaded again, preserving the computational benefits of cached prompts across model reloads.

## Background

### Router Mode Architecture

Router mode allows llama-server to dynamically load and unload multiple models on demand:
- Router process runs without any loaded model
- Each model instance runs as a separate child process
- Models are loaded/unloaded via HTTP API endpoints `/models/load` and `/models/unload`
- LRU eviction automatically unloads least-recently-used models when `models_max` limit is reached

### Current Prompt Cache Behavior

The server already implements an in-memory prompt cache (`server_prompt_cache`):
- Configured via `--cache-ram N` parameter (N MiB, 0 = disabled, -1 = unlimited)
- Stores KV cache state for processed prompts
- Automatically reuses cached prompts for similar inputs
- **Currently lost when model is unloaded**

### llama.cpp State Save/Load APIs

The core library provides comprehensive state management APIs:
- `llama_state_seq_save_file()` - Save single sequence state to file
- `llama_state_seq_load_file()` - Load single sequence state from file  
- `llama_state_save_file()` - Save full context state to file
- `llama_state_load_file()` - Load full context state from file

## Requirements

### Functional Requirements

1. **Save on Unload**: When a model is unloaded in router mode, save its prompt cache to disk
2. **Load on Reload**: When a model is reloaded, automatically load the saved prompt cache
3. **Opt-in Feature**: Feature must be explicitly enabled (disabled by default)
4. **Configuration**: Configurable via model preset INI file for router mode
5. **Invalid File Handling**: If saved file is invalid/corrupt, skip loading and overwrite on next save
6. **All Sequences**: Save all sequences in the prompt cache

### Non-Functional Requirements

1. **No Behavior Change**: Standard behavior must remain unchanged when feature is disabled
2. **Performance**: Minimal impact on load/unload operations
3. **Logging**: Comprehensive logging for debugging and monitoring
4. **Testing**: Unit tests for save/load functionality
5. **Code Quality**: Follow project conventions, include AI disclosure

## Implementation Plan

### Phase 1: Configuration Infrastructure

#### 1.1 Add Command-Line Argument

**File**: `common/arg.cpp`

Add new preset-only argument for KV cache persistence path:

```cpp
).set_env(COMMON_ARG_PRESET_KV_CACHE_PERSIST_PATH).set_preset_only());
```

**File**: `common/arg.h`

Define the environment variable constant:

```cpp
#define COMMON_ARG_PRESET_KV_CACHE_PERSIST_PATH "__PRESET_KV_CACHE_PERSIST_PATH"
```

**File**: `common/common.h`

Add field to `common_params` struct:

```cpp
std::string kv_cache_persist_path = ""; // path to save/load KV cache (router mode)
```

**Rationale**: Using preset-only argument ensures this is only configurable via INI files, not CLI, which is appropriate for router mode where models are defined in presets.

#### 1.2 Update Parameter Parsing

**File**: `common/arg.cpp`

Add argument handler:

```cpp
"--kv-cache-persist-path",
"PATH",
"Path to save/load KV cache state for router mode (empty = disabled)",
[](common_params & params, const std::string & value) {
    params.kv_cache_persist_path = value;
}
```

### Phase 2: Child Process KV Cache Save

#### 2.1 Add Save Functionality to Server Context

**File**: `tools/server/server-context.h`

Add method to `server_context`:

```cpp
bool save_kv_cache(const std::string & path);
```

**File**: `tools/server/server-context.cpp`

Implement `save_kv_cache()`:

```cpp
bool server_context::save_kv_cache(const std::string & path) {
    if (!impl || !impl->prompt_cache) {
        SRV_WRN("%s: prompt cache not available\n", __func__);
        return false;
    }
    
    auto & cache = impl->prompt_cache;
    if (cache->states.empty()) {
        SRV_INF("%s: prompt cache is empty, skipping save\n", __func__);
        return true;
    }
    
    SRV_INF("%s: saving %zu cached prompts to %s\n", __func__, cache->states.size(), path.c_str());
    
    // Create directory if needed
    auto dir_path = std::filesystem::path(path).parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        try {
            std::filesystem::create_directories(dir_path);
        } catch (const std::exception & e) {
            SRV_ERR("%s: failed to create directory %s: %s\n", __func__, dir_path.c_str(), e.what());
            return false;
        }
    }
    
    // Save each cached prompt as a sequence
    llama_context * ctx = get_llama_context();
    if (!ctx) {
        SRV_ERR("%s: llama context not available\n", __func__);
        return false;
    }
    
    int seq_id = 0;
    int saved_count = 0;
    
    for (const auto & state : cache->states) {
        if (state.data.empty()) {
            continue;
        }
        
        std::string seq_path = path + ".seq" + std::to_string(seq_id);
        
        size_t written = llama_state_seq_save_file(
            ctx,
            seq_path.c_str(),
            seq_id,
            state.tokens.get_text_tokens().data(),
            state.tokens.size()
        );
        
        if (written > 0) {
            saved_count++;
            SRV_INF("%s: saved sequence %d: %d tokens, %zu bytes to %s\n", 
                    __func__, seq_id, state.n_tokens(), written, seq_path.c_str());
        } else {
            SRV_WRN("%s: failed to save sequence %d\n", __func__, seq_id);
        }
        
        seq_id++;
    }
    
    SRV_INF("%s: saved %d/%zu sequences\n", __func__, saved_count, cache->states.size());
    return saved_count > 0;
}
```

**Rationale**: 
- Saves each cached prompt as a separate sequence file
- Creates parent directories if needed
- Logs detailed information for debugging
- Returns success if at least one sequence was saved

#### 2.2 Add Exit Signal Handler

**File**: `tools/server/server.cpp`

Modify the child server shutdown handler to save KV cache:

```cpp
// In main(), for child processes only
const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
if (router_port != nullptr) {
    // Save KV cache before shutdown if configured
    if (!params.kv_cache_persist_path.empty()) {
        SRV_INF("%s: saving KV cache to %s before shutdown\n", __func__, params.kv_cache_persist_path.c_str());
        ctx_server.save_kv_cache(params.kv_cache_persist_path);
    }
    
    monitor_thread = server_models::setup_child_server(shutdown_handler);
}
```

**Rationale**: Child processes receive exit signal from router, so we save KV cache before responding to the exit command.

### Phase 3: Child Process KV Cache Load

#### 3.1 Add Load Functionality to Server Context

**File**: `tools/server/server-context.h`

Add method to `server_context`:

```cpp
bool load_kv_cache(const std::string & path);
```

**File**: `tools/server/server-context.cpp`

Implement `load_kv_cache()`:

```cpp
bool server_context::load_kv_cache(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    
    SRV_INF("%s: attempting to load KV cache from %s\n", __func__, path.c_str());
    
    llama_context * ctx = get_llama_context();
    if (!ctx || !impl || !impl->prompt_cache) {
        SRV_ERR("%s: context or prompt cache not available\n", __func__);
        return false;
    }
    
    int loaded_count = 0;
    int seq_id = 0;
    
    while (true) {
        std::string seq_path = path + ".seq" + std::to_string(seq_id);
        
        if (!std::filesystem::exists(seq_path)) {
            SRV_INF("%s: no more sequence files found (last checked: %s)\n", __func__, seq_path.c_str());
            break;
        }
        
        std::vector<llama_token> tokens;
        size_t n_tokens = 0;
        
        size_t read = llama_state_seq_load_file(
            ctx,
            seq_path.c_str(),
            seq_id,
            tokens.data(),
            tokens.capacity(),
            &n_tokens
        );
        
        if (read > 0) {
            loaded_count++;
            SRV_INF("%s: loaded sequence %d: %zu tokens, %zu bytes from %s\n", 
                    __func__, seq_id, n_tokens, read, seq_path.c_str());
            
            // Add to prompt cache
            server_tokens stokens(tokens, false);
            server_prompt prompt;
            prompt.tokens = stokens;
            // Note: data will be populated by llama_state_seq_load_file
            impl->prompt_cache->states.push_back(std::move(prompt));
        } else {
            SRV_WRN("%s: failed to load sequence %d from %s, file may be invalid\n", 
                    __func__, seq_id, seq_path.c_str());
            // Continue trying other sequences
        }
        
        seq_id++;
    }
    
    SRV_INF("%s: loaded %d sequences into prompt cache\n", __func__, loaded_count);
    return loaded_count > 0;
}
```

**Rationale**:
- Loads sequence files sequentially (seq0, seq1, ...)
- Handles invalid files gracefully (logs warning and continues)
- Returns success if at least one sequence was loaded
- Adds loaded sequences to prompt cache for reuse

#### 3.2 Load After Model Initialization

**File**: `tools/server/server-context.cpp`

In `server_context_impl::load_model()`, after prompt cache initialization:

```cpp
if (params_base.cache_ram_mib != 0) {
    // ... existing prompt cache setup ...
    
    prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
    
    // Load saved KV cache if configured
    if (!params_base.kv_cache_persist_path.empty()) {
        load_kv_cache(params_base.kv_cache_persist_path);
    }
}
```

**Rationale**: Load KV cache immediately after prompt cache is created, before any requests are processed.

### Phase 4: Router Process Integration

#### 4.1 Pass KV Cache Path to Child Processes

**File**: `tools/server/server-models.cpp`

In `server_models::load()`, when spawning child process:

```cpp
// Add KV cache path to environment if configured
if (inst.meta.preset.get_option(COMMON_ARG_KV_CACHE_PERSIST_PATH, kv_cache_path)) {
    child_env.push_back("LLAMA_ARG_KV_CACHE_PERSIST_PATH=" + kv_cache_path);
}
```

**Rationale**: Environment variables are passed to child processes, allowing them to access the configuration.

#### 4.2 Document Configuration

**File**: `tools/server/README.md`

Add documentation for the new parameter:

```ini
# In model preset INI file
[model-name]
model = path/to/model.gguf
kv-cache-persist-path = /path/to/cache/model-name.kvcache
```

Add to parameter reference table:

| Parameter | Description |
|-----------|-------------|
| `kv-cache-persist-path` | Path prefix for saving/loading KV cache state in router mode (empty = disabled). Saves sequences as `{path}.seq0`, `{path}.seq1`, etc. |

### Phase 5: Logging and Error Handling

#### 5.1 Comprehensive Logging

Add detailed logging at key points:

1. **Configuration detected**: Log when KV cache persistence is enabled
2. **Save operation**: Log start, each sequence saved, total saved
3. **Load operation**: Log start, each sequence loaded, total loaded
4. **Errors**: Log directory creation failures, file I/O errors, invalid files
5. **Skip conditions**: Log when cache is empty, path is empty, etc.

#### 5.2 Error Handling Strategy

- **Directory creation failure**: Log error, continue without saving
- **File write failure**: Log warning, continue with next sequence
- **File read failure**: Log warning, continue with next sequence
- **Invalid file**: Log warning, continue with next sequence (will be overwritten on next save)
- **Empty cache**: Log info, skip save operation

### Phase 6: Testing

#### 6.1 Unit Tests

**File**: `tools/server/tests/unit/test_kv_cache_persistence.py`

Create comprehensive test suite:

```python
import pytest
from utils import *

def test_kv_cache_save_and_load():
    """Test basic KV cache save and load functionality"""
    # 1. Start server with KV cache persistence enabled
    # 2. Process a request to populate cache
    # 3. Unload model (should save cache)
    # 4. Reload model (should load cache)
    # 5. Verify cache is reused (check logs for cache hit)
    pass

def test_kv_cache_invalid_file():
    """Test handling of invalid cache files"""
    # 1. Create invalid cache file
    # 2. Start server with KV cache persistence enabled
    # 3. Verify server starts without error
    # 4. Process a request
    # 5. Verify invalid file is overwritten
    pass

def test_kv_cache_disabled():
    """Test that feature is disabled by default"""
    # 1. Start server without KV cache persistence
    # 2. Process a request
    # 3. Unload model
    # 4. Verify no cache files are created
    pass

def test_kv_cache_multiple_sequences():
    """Test saving/loading multiple cached prompts"""
    # 1. Process multiple different prompts
    # 2. Unload model
    # 3. Reload model
    # 4. Verify all sequences are loaded
    pass

def test_kv_cache_router_mode():
    """Test integration with router mode"""
    # 1. Start router server with KV cache persistence
    # 2. Load model via API
    # 3. Process request
    # 4. Unload model via API
    # 5. Reload model via API
    # 6. Verify cache is reused
    pass
```

#### 6.2 Integration Tests

**File**: `tools/server/tests/integration/test_kv_cache_router.py`

Test end-to-end router mode scenario:

```python
def test_router_kv_cache_lifecycle():
    """Test complete KV cache lifecycle in router mode"""
    # Start router server
    # Configure model with kv-cache-persist-path
    # Load model -> process request -> unload -> reload -> process similar request
    # Verify cache hit in logs
    pass
```

### Phase 7: Documentation

#### 7.1 User Documentation

**File**: `tools/server/README.md`

Add section "KV Cache Persistence":

```markdown
## KV Cache Persistence

When running in router mode, you can configure models to save their prompt cache (KV cache) to disk when unloaded and automatically reload it when the model is loaded again. This preserves the computational benefits of cached prompts across model reloads.

### Configuration

Add the `kv-cache-persist-path` parameter to your model preset:

```ini
[my-model]
model = Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
kv-cache-persist-path = /var/cache/llama/my-model.kvcache
```

### Behavior

- **Save**: When a model is unloaded, all cached prompts are saved to disk
- **Load**: When a model is loaded, saved cache is automatically restored
- **Invalid Files**: If saved files are invalid, they are skipped and overwritten on next save
- **Empty Cache**: If no prompts have been cached, no files are created

### File Format

Cache files are saved in llama.cpp's state format:
- `{path}.seq0`, `{path}.seq1`, etc. - One file per cached sequence
- Each file contains the KV cache state for a single prompt

### Example

```bash
# Start router server
./llama-server --models-preset models.ini

# Load model (will load saved cache if exists)
curl -X POST http://localhost:8080/models/load -d '{"model": "my-model"}'

# Process request (will use cached prompts if available)
curl -X POST http://localhost:8080/chat/completions -d '...'

# Unload model (will save cache)
curl -X POST http://localhost:8080/models/unload -d '{"model": "my-model"}'

# Reload model (will reload saved cache)
curl -X POST http://localhost:8080/models/load -d '{"model": "my-model"}'
```
```

#### 7.2 Developer Documentation

Add comments in code explaining:
- When KV cache is saved (on model unload/shutdown)
- When KV cache is loaded (on model load, after prompt cache init)
- File format and naming convention
- Error handling strategy

## Implementation Checklist

- [ ] Add `COMMON_ARG_PRESET_KV_CACHE_PERSIST_PATH` constant
- [ ] Add `kv_cache_persist_path` field to `common_params`
- [ ] Add CLI argument handler
- [ ] Implement `server_context::save_kv_cache()`
- [ ] Implement `server_context::load_kv_cache()`
- [ ] Add save call in child process exit handler
- [ ] Add load call after prompt cache initialization
- [ ] Pass KV cache path via environment variable
- [ ] Add comprehensive logging
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update README documentation
- [ ] Test with router mode
- [ ] Test with invalid cache files
- [ ] Test with empty cache
- [ ] Test with multiple sequences
- [ ] Verify no behavior change when disabled
- [ ] Run full test suite
- [ ] Format code with `git clang-format`
- [ ] Build successfully
- [ ] Create feature branch
- [ ] Commit with AI disclosure

## Branch Strategy

```bash
# Create feature branch
git checkout -b feature/kv-cache-persistence

# Make changes...
# Commit with AI disclosure:
git commit -m "[AI] Add KV cache persistence for router mode

Implement persistent KV cache saving and loading for router mode:
- Save prompt cache to disk on model unload
- Load saved cache on model reload
- Configurable via kv-cache-persist-path in model presets
- Opt-in feature, disabled by default
- Handle invalid files gracefully
- Comprehensive logging and tests"
```

## AI Disclosure

This implementation uses AI assistance for code generation. All commits will include `[AI]` prefix in the commit message, and code comments will be marked with `// [AI]` where appropriate, as required by the project's contribution guidelines.

## Risk Assessment

### Low Risk

- Feature is opt-in (disabled by default)
- No changes to core inference logic
- Graceful error handling for file I/O failures
- Invalid files are skipped, not fatal

### Medium Risk

- Disk space usage for cache files (mitigated by existing cache_ram limits)
- Potential for stale cache (mitigated by automatic overwrite)

### Mitigation Strategies

1. **Disk Space**: Cache files are bounded by `cache_ram_mib` limit
2. **Stale Cache**: Files are overwritten on each save
3. **Invalid Files**: Gracefully skipped and overwritten
4. **Performance**: Save/load operations are asynchronous with model load/unload

## Success Criteria

1. ✅ KV cache is saved when model is unloaded (if configured)
2. ✅ KV cache is loaded when model is reloaded (if configured)
3. ✅ Feature is disabled by default (no behavior change)
4. ✅ Invalid cache files are handled gracefully
5. ✅ Comprehensive logging for debugging
6. ✅ Unit tests pass
7. ✅ Integration tests pass
8. ✅ Documentation is complete
9. ✅ Code follows project conventions
10. ✅ Build succeeds without warnings

## Future Enhancements

1. **Compression**: Add optional compression for cache files
2. **Validation**: Add checksum validation for cache files
3. **Metadata**: Add timestamp and model metadata to cache files
4. **Automatic Cleanup**: Add option to automatically delete old cache files
5. **Remote Storage**: Support saving to remote storage (S3, etc.)
6. **Cache Statistics**: Add metrics for cache hit/miss rates

## References

- Router mode documentation: `tools/server/README.md`
- State save/load API: `include/llama.h` (lines 752-868)
- Prompt cache implementation: `tools/server/server-task.cpp` (lines 1358-1524)
- Example state save/load: `examples/save-load-state/save-load-state.cpp`
- Contribution guidelines: `CONTRIBUTING.md`
