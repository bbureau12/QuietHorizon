#!/usr/bin/env python3
"""
Test script for QuietHorizon MCP server.
Simulates how an AI assistant would use the MCP tools.
"""
import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.tools import (
    classify_audio_tool,
    batch_classify_tool,
    analyze_soundscape_tool,
    get_audio_info_tool
)


async def test_classify_audio():
    """Test classifying individual audio files."""
    print("=" * 60)
    print("TEST 1: Classify Audio (Cardinals)")
    print("=" * 60)
    
    result = await classify_audio_tool({
        "file_path": "quiet_horizon/test_data/240404__itinerantmonk108__northern-cardinal-closeup.wav"
    })
    
    print(result[0].text)
    
    print("\n" + "=" * 60)
    print("TEST 2: Classify Audio (Traffic)")
    print("=" * 60)
    
    result = await classify_audio_tool({
        "file_path": "quiet_horizon/test_data/691513__ania635__heavy_traffic_03.wav",
        "threshold": 0.5
    })
    
    print(result[0].text)


async def test_batch_classify():
    """Test batch classification."""
    print("\n" + "=" * 60)
    print("TEST 3: Batch Classify (test_data directory)")
    print("=" * 60)
    
    result = await batch_classify_tool({
        "directory": "quiet_horizon/test_data",
        "recursive": False,
        "threshold": 0.5
    })
    
    print(result[0].text)


async def test_analyze_soundscape():
    """Test soundscape analysis."""
    print("\n" + "=" * 60)
    print("TEST 4: Analyze Soundscape (Cardinal)")
    print("=" * 60)
    
    result = await analyze_soundscape_tool({
        "file_path": "quiet_horizon/test_data/240404__itinerantmonk108__northern-cardinal-closeup.wav",
        "include_spectrogram": False  # Skip image for cleaner output
    })
    
    print(result[0].text)
    
    print("\n" + "=" * 60)
    print("TEST 5: Analyze Soundscape (Traffic)")
    print("=" * 60)
    
    result = await analyze_soundscape_tool({
        "file_path": "quiet_horizon/test_data/691513__ania635__heavy_traffic_03.wav",
        "include_spectrogram": False
    })
    
    print(result[0].text)


async def test_audio_info():
    """Test audio metadata extraction."""
    print("\n" + "=" * 60)
    print("TEST 6: Get Audio Info (Cardinal)")
    print("=" * 60)
    
    result = await get_audio_info_tool({
        "file_path": "quiet_horizon/test_data/240404__itinerantmonk108__northern-cardinal-closeup.wav"
    })
    
    print(result[0].text)
    
    print("\n" + "=" * 60)
    print("TEST 7: Get Audio Info (Traffic)")
    print("=" * 60)
    
    result = await get_audio_info_tool({
        "file_path": "quiet_horizon/test_data/691513__ania635__heavy_traffic_03.wav"
    })
    
    print(result[0].text)


async def main():
    """Run all MCP tool tests."""
    print("\n" + "[MCP] QuietHorizon MCP Server Tool Testing")
    print("Testing with real audio files from quiet_horizon/test_data/\n")
    
    try:
        # Test individual tools
        await test_classify_audio()
        await test_batch_classify()
        await test_analyze_soundscape()
        await test_audio_info()
        
        print("\n" + "=" * 60)
        print("[PASS] All MCP tool tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[FAIL] Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
