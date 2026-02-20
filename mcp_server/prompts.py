"""MCP Prompt templates for QuietHorizon audio analysis."""

import json
from pathlib import Path
from typing import Any

from mcp.types import PromptMessage, TextContent


async def classification_report_prompt(arguments: dict[str, str]) -> PromptMessage:
    """
    Generate a detailed environmental audio classification report.
    
    Args:
        arguments: Dict with 'file_path'
        
    Returns:
        PromptMessage with classification report template
    """
    file_path = arguments.get("file_path", "<audio_file.wav>")
    
    prompt_text = f"""Please analyze the audio file and provide a comprehensive environmental sound classification report.

Audio File: {file_path}

Use the classify_audio and analyze_soundscape tools to gather information, then create a report including:

1. **Classification Summary**
   - Primary classification (Nature vs Anthropogenic)
   - Confidence level and probabilities
   - Threshold used

2. **Acoustic Analysis**
   - Spectral characteristics (frequency content)
   - Temporal patterns (duration, rhythm)
   - Energy profile

3. **Ecological Context** (if nature)
   - Likely source (bird, mammal, weather, etc.)
   - Habitat implications
   - Biodiversity indicators

4. **Environmental Assessment**
   - Sound quality rating
   - Anthropogenic disturbance level
   - Recommendations for monitoring

5. **Technical Metadata**
   - Audio format and duration
   - Sample rate and quality
   - Processing details

Format the report in clear sections with bullet points and technical details.
"""
    
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=prompt_text)
    )


async def batch_analysis_prompt(arguments: dict[str, str]) -> PromptMessage:
    """
    Analyze multiple audio files and create comparative summary.
    
    Args:
        arguments: Dict with 'directory'
        
    Returns:
        PromptMessage with batch analysis template
    """
    directory = arguments.get("directory", "<audio_directory/>")
    
    prompt_text = f"""Please analyze all audio files in the specified directory and create a comparative soundscape report.

Directory: {directory}

Use the batch_classify tool to process all files, then create a summary including:

1. **Overview**
   - Total files processed
   - Success/failure rate
   - Overall nature vs anthropogenic distribution

2. **Classification Breakdown**
   - Percentage of nature sounds
   - Percentage of anthropogenic sounds
   - Confidence distribution

3. **Individual File Results**
   - Table with filename, classification, confidence
   - Notable high/low confidence cases
   - Any processing errors

4. **Soundscape Assessment**
   - Overall environmental quality
   - Dominant sound types
   - Temporal patterns (if timestamps in filenames)
   - Areas of concern (high anthropogenic presence)

5. **Recommendations**
   - Monitoring priorities
   - Files requiring manual review
   - Further analysis suggestions

Present results in a structured format with visualizations if possible (tables, percentages).
"""
    
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=prompt_text)
    )


async def soundscape_summary_prompt(arguments: dict[str, str]) -> PromptMessage:
    """
    Create ecological soundscape assessment with recommendations.
    
    Args:
        arguments: Dict with 'file_path'
        
    Returns:
        PromptMessage with soundscape assessment template
    """
    file_path = arguments.get("file_path", "<soundscape.wav>")
    
    prompt_text = f"""Please create an ecological soundscape assessment for the provided audio recording.

Audio File: {file_path}

Use classify_audio and analyze_soundscape tools, then generate:

1. **Soundscape Characterization**
   - Primary sound classification
   - Acoustic complexity (spectral features)
   - Sound level and variability

2. **Ecological Indicators**
   - Biophony: Biological sound presence
   - Geophony: Natural environmental sounds
   - Anthropophony: Human-generated noise
   - Overall naturalness score

3. **Habitat Quality Assessment**
   - Based on sound diversity and type
   - Anthropogenic disturbance level
   - Acoustic habitat suitability

4. **Conservation Implications**
   - Biodiversity indicators from sounds
   - Noise pollution assessment
   - Habitat health status

5. **Monitoring Recommendations**
   - Optimal recording times
   - Priority species to monitor
   - Baseline establishment suggestions
   - Mitigation strategies (if needed)

Provide actionable insights for ecologists and conservationists.
"""
    
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=prompt_text)
    )
