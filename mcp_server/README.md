# QuietHorizon MCP Server

**Model Context Protocol server for environmental audio classification**

Connect AI assistants like Claude Desktop to QuietHorizon's CNN model for automated audio analysis, batch processing, and ecological monitoring through standardized MCP tools.

## Features

### 🛠️ Tools

- **`classify_audio`** - Classify single audio files (nature vs anthropogenic)
- **`batch_classify`** - Process entire directories with aggregate statistics
- **`analyze_soundscape`** - Detailed spectral analysis with visualizations
- **`get_audio_info`** - Extract audio metadata and technical details

### 📚 Resources

- **`model://info`** - QuietHorizon CNN model specifications and performance
- **`dataset://statistics`** - Training dataset composition and characteristics
- **`config://formats`** - Supported audio formats and limits

### 💬 Prompts

- **`classification_report`** - Generate comprehensive environmental audio reports
- **`batch_analysis`** - Create comparative soundscape assessments
- **`soundscape_summary`** - Ecological monitoring recommendations

## Quick Start

### 1. Installation

```powershell
# From QuietHorizon root directory
cd mcp_server
pip install -e .
```

### 2. Claude Desktop Configuration

Add to your Claude Desktop config file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "quiethorizon": {
      "command": "python",
      "args": [
        "D:\\Projects\\QuietHorizon\\mcp_server\\server.py"
      ],
      "env": {
        "PYTHONPATH": "D:\\Projects\\QuietHorizon"
      }
    }
  }
}
```

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "quiethorizon": {
      "command": "python3",
      "args": [
        "/path/to/QuietHorizon/mcp_server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/QuietHorizon"
      }
    }
  }
}
```

### 3. Restart Claude Desktop

The QuietHorizon tools will appear in Claude's tool menu (🔌 icon).

## Usage Examples

### Example 1: Classify a Single Audio File

**In Claude Desktop:**
```
Classify this audio file: C:\recordings\bird_morning.wav
```

**Claude will:**
1. Use `classify_audio` tool
2. Return nature/anthro classification with confidence
3. Provide acoustic interpretation

### Example 2: Batch Process Directory

```
Analyze all audio files in C:\field_recordings\site_A\
```

**Claude will:**
1. Use `batch_classify` tool
2. Process all supported formats recursively
3. Generate summary statistics (% nature vs anthro)
4. Highlight notable classifications

### Example 3: Detailed Soundscape Analysis

```
Create an ecological assessment of soundscape_dawn.flac
```

**Claude will:**
1. Use `analyze_soundscape` tool
2. Extract spectral features
3. Generate classification with acoustic context
4. Provide conservation insights

### Example 4: Use Prompts

```
Use the classification_report prompt for forest_recording.wav
```

**Claude will:**
1. Execute the pre-built prompt template
2. Gather classification and spectral data
3. Generate comprehensive formatted report
4. Include recommendations

## Tool Reference

### classify_audio

```json
{
  "file_path": "path/to/audio.wav",
  "threshold": 0.5
}
```

**Returns:**
```json
{
  "file": "audio.wav",
  "classification": "nature",
  "confidence": 0.9234,
  "probabilities": {
    "nature": 0.9234,
    "anthropogenic": 0.0766
  },
  "threshold": 0.5,
  "audio_info": {
    "duration_seconds": 12.5,
    "sample_rate": 22050
  }
}
```

### batch_classify

```json
{
  "directory": "path/to/recordings",
  "recursive": true,
  "threshold": 0.5
}
```

**Returns:**
```json
{
  "directory": "path/to/recordings",
  "total_files": 45,
  "processed": 43,
  "failed": 2,
  "summary": {
    "nature": 31,
    "anthropogenic": 12,
    "nature_percentage": 72.1
  },
  "results": [...]
}
```

### analyze_soundscape

```json
{
  "file_path": "path/to/soundscape.wav",
  "include_spectrogram": true
}
```

**Returns:**
- Spectral features (centroid, rolloff, zero-crossing rate)
- Energy profile (RMS)
- Classification probabilities
- Mel-spectrogram image (if requested)

### get_audio_info

```json
{
  "file_path": "path/to/audio.mp3"
}
```

**Returns:**
```json
{
  "file": "audio.mp3",
  "format": ".mp3",
  "size_mb": 3.4,
  "duration_seconds": 45.2,
  "sample_rate": 44100,
  "channels": 2
}
```

## Resource Reference

### model://info

Returns QuietHorizon CNN specifications:
- Architecture details
- Performance metrics (~95% accuracy)
- Input/output formats
- Preprocessing parameters

### dataset://statistics

Returns training dataset composition:
- Class distribution (nature vs anthro)
- Example categories (60+ nature types, 3 anthro types)
- Acoustic characteristics
- Data augmentation techniques

### config://formats

Returns supported formats and limits:
- Supported extensions (.wav, .mp3, .flac, .ogg, .m4a)
- File size limits (50 MB max)
- Recommended formats (WAV/FLAC lossless)
- Preprocessing details

## Advanced Usage

### Custom Thresholds

Adjust classification sensitivity:

```
Classify bird_call.wav with a strict threshold of 0.7 for nature
```

QuietHorizon will require 70% confidence for "nature" classification.

### Recursive Directory Search

```
Batch classify all audio in C:\monitoring\ including subdirectories
```

Will search recursively through all folders.

### Spectrogram Visualization

```
Show me the spectrogram for whale_song.wav
```

Returns mel-spectrogram image with acoustic analysis.

## Integration with Other Tools

### VS Code + Cline

Configure in Cline MCP settings, then:

```
@mcp quiethorizon classify my_recording.wav
```

### Automation Scripts

Call MCP server directly from Python:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["mcp/server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        result = await session.call_tool(
            "classify_audio",
            arguments={"file_path": "test.wav"}
        )
```

## Supported Audio Formats

✅ **Recommended (Lossless):**
- WAV - Waveform Audio
- FLAC - Free Lossless Audio Codec

⚠️ **Supported (Lossy):**
- MP3 - MPEG Audio Layer III
- OGG - Ogg Vorbis
- M4A - MPEG-4 Audio

**Note:** Lossy formats may reduce classification accuracy due to compression artifacts.

## Performance

- **Single file classification:** ~1-2 seconds
- **Batch processing:** ~2 seconds per file
- **Spectrogram generation:** ~0.5 seconds
- **Model loading:** ~3 seconds (first run only)

Model is cached after first invocation for faster subsequent calls.

## Troubleshooting

### Server Not Appearing in Claude Desktop

1. Check config file syntax (valid JSON)
2. Use absolute paths in configuration
3. Verify Python path is correct (`which python`)
4. Check Claude logs: `%APPDATA%\Claude\logs\`

### Classification Errors

1. Ensure audio file format is supported
2. Check file size (<50 MB)
3. Verify file is not corrupted
4. Try converting to WAV format

### Import Errors

1. Ensure PYTHONPATH includes QuietHorizon root
2. Verify all dependencies installed: `pip install -e .`
3. Check that QuietHorizon model is downloaded

### Model Download Issues

Model auto-downloads from HuggingFace on first run. If fails:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="bbureau12/QuietHorizon",
    filename="quiet_horizon_cnn.keras"
)
```

## Development

### Running Tests

```powershell
pip install -e ".[dev]"
pytest tests/
```

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Logs include:
- Tool execution timing
- Model loading status
- Audio processing steps
- Error stack traces

### Adding Custom Tools

1. Create function in `tools.py`
2. Add tool definition in `server.py` `list_tools()`
3. Register in `call_tool()` dispatcher
4. Update documentation

## Architecture

```
mcp_server/
├── server.py       # Main MCP server (FastMCP)
├── tools.py        # Tool implementations
├── resources.py    # Resource handlers
├── prompts.py      # Prompt templates
├── pyproject.toml  # Dependencies
└── README.md       # This file
```

**Integration:**
- Uses `quiet_horizon.audio.preprocessing` for audio processing
- Uses `quiet_horizon.inference_cnn` for model loading
- Lazy-loads model (singleton pattern)
- Async/await for non-blocking operations

## License

MIT License - see root LICENSE file

## Contributing

See root CONTRIBUTING.md for guidelines.

## Support

- **Issues:** GitHub Issues
- **Documentation:** See root README.md for QuietHorizon details
- **MCP Protocol:** https://modelcontextprotocol.io/

---

**Built with:** [Model Context Protocol](https://modelcontextprotocol.io/) | **Powered by:** QuietHorizon CNN
