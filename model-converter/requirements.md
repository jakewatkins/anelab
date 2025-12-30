# Model Converter

## Overview
A Python command-line tool that downloads models from HuggingFace and converts them to Apple's Core ML (.mlmodel) format for use in iOS, macOS, and other Apple platforms.

## Core Functionality
- Download models from HuggingFace Hub using model identifiers
- Convert downloaded models to Core ML format (.mlmodel)
- Save converted models to the current working directory
- Command-line interface with single parameter input

## Usage
```bash
python model-converter.py <huggingface_model_name>
```

### Example
```bash
python model-converter.py mlx-community/mistralai_Ministral-3-14B-Instruct-2512-MLX-MXFP4
```

## Technical Requirements

### Dependencies
- **Python**: 3.8+ (required for Core ML Tools)
- **coremltools**: Apple's library for Core ML model conversion
- **transformers**: HuggingFace transformers library for model loading
- **torch**: PyTorch framework (required by transformers)
- **huggingface_hub**: For downloading models from HuggingFace Hub

### Supported Model Types
- Text generation models (GPT, LLaMA, Mistral, etc.)
- BERT-style encoder models
- Vision transformers (if applicable)
- Other transformer architectures supported by HuggingFace transformers

### Input Validation
- Verify HuggingFace model name format (user/model-name or organization/model-name)
- Check model availability on HuggingFace Hub
- Validate model compatibility with Core ML conversion
- Handle authentication for private models (if needed)

### Output Specifications
- **File Format**: .mlmodel (Core ML package format)
- **File Location**: Current working directory
- **Naming Convention**: Use original model name with .mlmodel extension
- **Metadata**: Preserve model information (author, description, license)

### Error Handling
- Network connectivity issues
- Invalid model names
- Unsupported model architectures
- Insufficient disk space
- Memory limitations during conversion
- HuggingFace Hub authentication errors

### Performance Considerations
- Large model handling (memory management)
- Download progress indication
- Conversion progress feedback
- Temporary file cleanup after conversion

### Platform Support
- **Primary**: macOS (native Core ML support)
- **Secondary**: Linux/Windows (for development and testing)

## Implementation Details

### Workflow
1. Parse command-line arguments
2. Validate HuggingFace model identifier
3. Download model files from HuggingFace Hub
4. Load model using transformers library
5. Convert model to Core ML format using coremltools
6. Save .mlmodel file to current directory
7. Clean up temporary files
8. Provide success/failure feedback

### Configuration
- Default model precision (float16/float32)
- Maximum model size limits
- Timeout settings for downloads
- Cache directory for downloaded models

### Logging
- Download progress
- Conversion status
- Error messages with context
- Performance metrics (optional)

## Limitations
- **No Compilation**: Tool does not compile models for specific devices
- **No Optimization**: Basic conversion without device-specific optimizations
- **Memory Constraints**: Large models may require significant RAM
- **Format Support**: Limited to models supported by both HuggingFace and Core ML

## Future Enhancements (Out of Scope)
- Batch conversion of multiple models
- Custom optimization parameters
- Model compilation for specific devices
- GUI interface
- Model validation and testing
- Integration with Xcode projects  