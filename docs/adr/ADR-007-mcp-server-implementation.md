# ADR-007: Model Context Protocol (MCP) Server Implementation

**Status:** Accepted  
**Date:** 2026-02-19  
**Deciders:** Development Team  
**Tags:** integration, mcp, api, ml-inference

## Context

QuietHorizon provides audio classification capabilities through a CLI and web frontend, but there's no programmatic interface for automation or integration with AI assistants. The Model Context Protocol (MCP) is an emerging standard for connecting AI assistants to external tools and data sources, offering a standardized way to expose ML capabilities.

### Use Cases
- AI assistants (Claude Desktop, IDEs) accessing audio classification
- Automated batch processing workflows
- Integration with ecological monitoring systems
- Standardized tool interface for developers

### Current Limitations
- No programmatic API beyond CLI
- Manual file-by-file processing required
- Limited integration with AI workflows
- No standardized tool interface

## Decision

We will implement a Model Context Protocol (MCP) server that exposes QuietHorizon's classification capabilities through standardized tools, resources, and prompts.

### Implementation Scope

**1. Tools (4)**
- `classify_audio` - Single file classification
- `batch_classify` - Directory-based batch processing
- `analyze_soundscape` - Detailed spectral analysis
- `get_audio_info` - Audio metadata extraction

**2. Resources (3)**
- `model://info` - Model specifications and performance
- `dataset://statistics` - Training data composition
- `config://formats` - Supported formats and limits

**3. Prompts (3)**
- `classification_report` - Comprehensive audio analysis template
- `batch_analysis` - Multi-file comparative assessment
- `soundscape_summary` - Ecological monitoring report

### Architecture
```
mcp_server/
├── server.py       # FastMCP server implementation
├── tools.py        # Tool function implementations
├── resources.py    # Resource data providers
├── prompts.py      # Prompt template generators
├── pyproject.toml  # MCP-specific dependencies
└── README.md       # MCP server documentation
```

### Technology Stack
- **MCP SDK:** `mcp>=0.9.0` (official Python SDK)
- **Communication:** stdio (standard input/output)
- **Execution:** Async/await for non-blocking operations
- **Model Integration:** Reuses `quiet_horizon.audio` and `quiet_horizon.inference_cnn`

## Rationale

### Why MCP?

**Standardization**
- Industry-standard protocol backed by Anthropic
- Consistent interface across AI assistants and tools
- Future-proof as MCP adoption grows

**Developer Experience**
- Simple JSON-based tool definitions
- Standard client libraries available
- Well-documented protocol specification

**Integration Potential**
- Claude Desktop (immediate use case)
- VS Code + Cline integration
- Custom automation scripts
- Future MCP-compatible tools

### Why NOT a REST API?

**MCP Advantages:**
- ✅ No server/port management required
- ✅ Process-based isolation
- ✅ Built-in tool discovery
- ✅ Standardized error handling
- ✅ Native AI assistant integration

**REST API Would Require:**
- ❌ HTTP server setup (FastAPI/Flask)
- ❌ Authentication/authorization
- ❌ CORS configuration
- ❌ Custom API documentation
- ❌ Service deployment/hosting

**Verdict:** MCP provides better developer experience for AI integration with lower operational overhead.

### Alternative: Direct Python Package

**Considered:** Exposing Python functions directly  
**Rejected Because:**
- Requires Python environment integration
- No standardized tool interface
- Limited to Python ecosystems
- No AI assistant discovery
- Manual prompt engineering needed

## Consequences

### Positive

**Expanded Use Cases**
- AI assistants can classify audio conversationally
- Automated ecological monitoring workflows
- Batch processing via AI-driven scripts
- Standardized integration point

**Improved Developer Experience**
- No custom API learning curve
- Standard MCP client libraries work
- Tool auto-discovery in compatible clients
- Pre-built prompt templates reduce setup

**Future-Proofing**
- MCP adoption growing rapidly
- Multiple client implementations
- Protocol versioning and compatibility
- Community ecosystem developing

**Resource Efficiency**
- Model loaded once, reused across calls
- Process-based isolation prevents conflicts
- No always-on server required
- Lazy loading reduces startup time

### Negative

**New Dependency**
- Requires `mcp` package (~minimal size)
- MCP protocol learning curve for contributors
- Additional testing surface area

**Limited to MCP Clients**
- Only works with MCP-compatible tools
- Web browsers can't directly consume
- Non-MCP automation needs different approach
- Protocol version lock-in

**Async Complexity**
- All tools must be async functions
- Requires asyncio understanding
- Potential blocking operation issues
- Error handling more complex

### Mitigation Strategies

**For MCP Dependency:**
- Keep MCP server isolated in `mcp_server/` directory
- Document installation separately
- Make it optional (not in core `setup.py`)

**For Limited Clients:**
- Maintain CLI for direct usage
- Keep Streamlit frontend for web access
- Document REST API option for future
- MCP can call CLI internally if needed

**For Async Complexity:**
- Use `asyncio.to_thread()` for blocking calls
- Clear error handling patterns
- Comprehensive logging
- Type hints for tool signatures

## Implementation Details

### Tool Design Principles

**1. Single Responsibility**
- Each tool does one thing well
- Clear input/output contracts
- Predictable error handling

**2. Idempotency**
- Same input → same output
- No side effects (except logging)
- Safe to retry

**3. Informative Responses**
- Structured JSON output
- Include confidence/metadata
- Clear error messages

### Resource Caching

Resources are static and cached:
- `model://info` - Hardcoded metadata
- `dataset://statistics` - Static documentation
- `config://formats` - Configuration snapshot

No database or dynamic queries needed.

### Prompt Templates

Prompts leverage tools internally:
- `classification_report` → calls `classify_audio` + `analyze_soundscape`
- `batch_analysis` → calls `batch_classify`
- `soundscape_summary` → combines multiple tools

AI assistant executes tools and formats results.

### Security Considerations

**File Access:**
- Tools accept file paths from user
- No path validation/sandboxing by default
- **Mitigation:** Document trusted execution context, add path validation if needed

**Model Access:**
- Model file loaded from local/HF cache
- No remote code execution
- **Mitigation:** Integrity checks via HF checksums

**Resource Limits:**
- File size limits enforced (50 MB)
- No rate limiting by default
- **Mitigation:** Document for shared deployment, add limits if needed

## Testing Strategy

**Unit Tests:**
- Mock MCP server responses
- Test tool functions independently
- Validate resource JSON schemas

**Integration Tests:**
- Test full MCP tool calls
- Verify Claude Desktop integration
- Batch processing workflows

**Manual Testing:**
- Claude Desktop end-to-end
- Multiple audio formats
- Error scenarios

## Performance Expectations

- **First invocation:** ~3-5 seconds (model load)
- **Subsequent calls:** ~1-2 seconds per file
- **Batch processing:** ~2 seconds per file
- **Resource queries:** <100ms (cached)

Model singleton ensures loading once per server lifetime.

## Documentation

- `mcp_server/README.md` - Complete MCP server guide
- Root README.md - Link to MCP capabilities
- ADR-007 (this document) - Design rationale
- Code comments - Function-level documentation

## Rollout Plan

1. ✅ Implement MCP server structure
2. ✅ Create tools, resources, prompts
3. ✅ Write comprehensive documentation
4. ⏳ Test with Claude Desktop
5. ⏳ Add MCP section to root README
6. ⏳ Create demo video/screenshots
7. ⏳ Publish on MCP community registry

## Future Enhancements

**Additional Tools:**
- `compare_audio` - Side-by-side comparison
- `export_report` - Generate PDF/HTML reports
- `monitor_directory` - Watch folder for new files
- `train_on_feedback` - Active learning pipeline

**Additional Resources:**
- `classification://history` - Recent predictions log
- `performance://metrics` - Real-time statistics
- `model://versions` - Model version management

**Authentication:**
- API key validation (if needed)
- Rate limiting per client
- Usage quotas

**Streaming:**
- Real-time audio classification
- Progress updates for batch jobs
- Live monitoring integrations

## References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop MCP Guide](https://modelcontextprotocol.io/quickstart/server)
- QuietHorizon ADR-001: Audio Preprocessing Module
- QuietHorizon ADR-004: Model Distribution via HuggingFace

## Approval

**Decision:** Implement MCP server as described  
**Review:** Architecture validated  
**Status:** Implementation complete, testing in progress
