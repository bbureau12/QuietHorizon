#!/usr/bin/env python3
"""
QuietHorizon MCP Server

Exposes audio classification capabilities via the Model Context Protocol.
Allows AI assistants to classify environmental sounds through standardized tools,
resources, and prompts.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

from tools import (
    classify_audio_tool,
    batch_classify_tool,
    analyze_soundscape_tool,
    get_audio_info_tool,
)
from resources import (
    get_model_info,
    get_dataset_statistics,
    get_supported_formats,
)
from prompts import (
    classification_report_prompt,
    batch_analysis_prompt,
    soundscape_summary_prompt,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("quiethorizon-mcp")

# Initialize MCP server
app = Server("quiethorizon-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available audio classification tools."""
    return [
        Tool(
            name="classify_audio",
            description="Classify a single audio file as nature or anthropogenic sound. "
                       "Returns probability scores and confidence level.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file (WAV, MP3, FLAC, OGG, M4A)"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Classification threshold (0-1, default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="batch_classify",
            description="Classify multiple audio files from a directory. "
                       "Processes all supported audio formats and returns aggregated results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to directory containing audio files"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search subdirectories recursively",
                        "default": False
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Classification threshold (0-1, default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["directory"]
            }
        ),
        Tool(
            name="analyze_soundscape",
            description="Perform detailed spectral analysis of an audio file. "
                       "Returns mel-spectrogram visualization and acoustic features.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file"
                    },
                    "include_spectrogram": {
                        "type": "boolean",
                        "description": "Include spectrogram image in response",
                        "default": True
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="get_audio_info",
            description="Get technical information about an audio file "
                       "(duration, sample rate, channels, format).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file"
                    }
                },
                "required": ["file_path"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Execute a tool with the given arguments."""
    try:
        if name == "classify_audio":
            return await classify_audio_tool(arguments)
        elif name == "batch_classify":
            return await batch_classify_tool(arguments)
        elif name == "analyze_soundscape":
            return await analyze_soundscape_tool(arguments)
        elif name == "get_audio_info":
            return await get_audio_info_tool(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Tool execution failed: {name}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="model://info",
            name="Model Information",
            description="QuietHorizon CNN model metadata, accuracy, and training details",
            mimeType="application/json"
        ),
        Resource(
            uri="dataset://statistics",
            name="Dataset Statistics",
            description="Training dataset composition and class distribution",
            mimeType="application/json"
        ),
        Resource(
            uri="config://formats",
            name="Supported Audio Formats",
            description="List of supported audio file formats and size limits",
            mimeType="application/json"
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    try:
        if uri == "model://info":
            return await get_model_info()
        elif uri == "dataset://statistics":
            return await get_dataset_statistics()
        elif uri == "config://formats":
            return await get_supported_formats()
        else:
            raise ValueError(f"Unknown resource: {uri}")
    except Exception as e:
        logger.error(f"Resource read failed: {uri}", exc_info=True)
        return f'{{"error": "{str(e)}"}}'


@app.list_prompts()
async def list_prompts() -> list[dict]:
    """List available prompt templates."""
    return [
        {
            "name": "classification_report",
            "description": "Generate a detailed environmental audio classification report",
            "arguments": [
                {
                    "name": "file_path",
                    "description": "Path to audio file to analyze",
                    "required": True
                }
            ]
        },
        {
            "name": "batch_analysis",
            "description": "Analyze multiple audio files and create comparative summary",
            "arguments": [
                {
                    "name": "directory",
                    "description": "Directory containing audio files",
                    "required": True
                }
            ]
        },
        {
            "name": "soundscape_summary",
            "description": "Create ecological soundscape assessment with recommendations",
            "arguments": [
                {
                    "name": "file_path",
                    "description": "Path to soundscape recording",
                    "required": True
                }
            ]
        },
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str]) -> str:
    """Get a prompt template with arguments filled in."""
    try:
        if name == "classification_report":
            return await classification_report_prompt(arguments)
        elif name == "batch_analysis":
            return await batch_analysis_prompt(arguments)
        elif name == "soundscape_summary":
            return await soundscape_summary_prompt(arguments)
        else:
            raise ValueError(f"Unknown prompt: {name}")
    except Exception as e:
        logger.error(f"Prompt generation failed: {name}", exc_info=True)
        return f"Error: {str(e)}"


async def main():
    """Run the MCP server."""
    logger.info("Starting QuietHorizon MCP Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
