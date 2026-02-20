#!/usr/bin/env python3
"""
Validation script for QuietHorizon MCP server.
Tests imports, structure, and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all MCP modules can be imported."""
    print("Testing imports...")
    try:
        import mcp_server.server
        import mcp_server.tools
        import mcp_server.resources
        import mcp_server.prompts
        print("[PASS] All modules imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_server_structure():
    """Test that server has required components."""
    print("\nTesting server structure...")
    try:
        from mcp_server.server import app, list_tools, call_tool, list_resources, read_resource
        
        # Check server is initialized
        assert app is not None, "Server app not initialized"
        print("âœ“ Server app initialized")
        
        # Check handlers are defined
        assert callable(list_tools), "list_tools not callable"
        assert callable(call_tool), "call_tool not callable"
        assert callable(list_resources), "list_resources not callable"
        assert callable(read_resource), "read_resource not callable"
        print("âœ“ All handlers defined")
        
        return True
    except Exception as e:
        print(f"âœ— Structure test failed: {e}")
        return False


def test_tool_functions():
    """Test that tool functions are defined."""
    print("\nTesting tool functions...")
    try:
        from mcp_server.tools import (
            classify_audio_tool,
            batch_classify_tool,
            analyze_soundscape_tool,
            get_audio_info_tool
        )
        
        tools = [
            ("classify_audio", classify_audio_tool),
            ("batch_classify", batch_classify_tool),
            ("analyze_soundscape", analyze_soundscape_tool),
            ("get_audio_info", get_audio_info_tool),
        ]
        
        for name, func in tools:
            assert callable(func), f"{name} not callable"
            print(f"âœ“ {name} defined")
        
        return True
    except Exception as e:
        print(f"âœ— Tool function test failed: {e}")
        return False


def test_resource_functions():
    """Test that resource functions are defined."""
    print("\nTesting resource functions...")
    try:
        from mcp_server.resources import (
            get_model_info,
            get_dataset_statistics,
            get_supported_formats
        )
        
        resources = [
            ("get_model_info", get_model_info),
            ("get_dataset_statistics", get_dataset_statistics),
            ("get_supported_formats", get_supported_formats),
        ]
        
        for name, func in resources:
            assert callable(func), f"{name} not callable"
            print(f"âœ“ {name} defined")
        
        return True
    except Exception as e:
        print(f"âœ— Resource function test failed: {e}")
        return False


def test_prompt_functions():
    """Test that prompt functions are defined."""
    print("\nTesting prompt functions...")
    try:
        from mcp_server.prompts import (
            classification_report_prompt,
            batch_analysis_prompt,
            soundscape_summary_prompt
        )
        
        prompts = [
            ("classification_report", classification_report_prompt),
            ("batch_analysis", batch_analysis_prompt),
            ("soundscape_summary", soundscape_summary_prompt),
        ]
        
        for name, func in prompts:
            assert callable(func), f"{name} not callable"
            print(f"âœ“ {name} defined")
        
        return True
    except Exception as e:
        print(f"âœ— Prompt function test failed: {e}")
        return False


def test_quiethorizon_integration():
    """Test that QuietHorizon modules can be imported."""
    print("\nTesting QuietHorizon integration...")
    try:
        from quiet_horizon.audio.preprocessing import audio_to_spectrogram
        from quiet_horizon.inference_cnn import load_model
        
        print("âœ“ QuietHorizon modules accessible")
        return True
    except ImportError as e:
        print(f"âœ— QuietHorizon import failed: {e}")
        print("  Note: This is expected if running outside QuietHorizon root")
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("QuietHorizon MCP Server Validation")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_server_structure,
        test_tool_functions,
        test_resource_functions,
        test_prompt_functions,
        test_quiethorizon_integration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("âœ… All validations passed! MCP server is ready.")
        return 0
    else:
        print("âš ï¸  Some validations failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
