#!/bin/bash

# Model Converter Setup Script
# This script helps resolve common compatibility issues

echo "🔧 Model Converter Setup & Diagnostics"
echo "======================================"

# Check Python version
echo "🐍 Python Version Check:"
python_version=$(python3 --version 2>&1)
echo "   Current: $python_version"

# Extract version numbers
python_major=$(python3 -c "import sys; print(sys.version_info.major)")
python_minor=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 14 ]; then
    echo "   ⚠️  WARNING: Python 3.14+ may have compatibility issues with CoreML Tools"
    echo "   💡 Recommendation: Use Python 3.9-3.11 for best compatibility"
fi

# Check platform
echo ""
echo "🖥️  Platform Check:"
os=$(uname -s)
arch=$(uname -m)
echo "   OS: $os"
echo "   Architecture: $arch"

if [ "$os" != "Darwin" ]; then
    echo "   ⚠️  WARNING: Core ML is optimized for macOS"
    echo "   💡 Limited functionality expected on non-macOS platforms"
fi

# Check if we're in a virtual environment
echo ""
echo "🏠 Environment Check:"
if [ -n "$VIRTUAL_ENV" ]; then
    echo "   ✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "   ⚠️  No virtual environment detected"
    echo "   💡 Recommendation: Create and activate a virtual environment"
    echo "      python3 -m venv venv"
    echo "      source venv/bin/activate"
fi

# Function to install with fallback options
install_packages() {
    echo ""
    echo "📦 Installing Packages..."
    
    # Try pip first
    echo "   Attempting pip install..."
    if pip install -r requirements.txt; then
        echo "   ✅ Pip install successful"
        return 0
    else
        echo "   ❌ Pip install failed"
    fi
    
    # Try conda if available
    if command -v conda &> /dev/null; then
        echo "   Attempting conda install..."
        conda install -c conda-forge pytorch transformers huggingface_hub
        conda install -c apple coremltools
        if [ $? -eq 0 ]; then
            echo "   ✅ Conda install successful"
            return 0
        else
            echo "   ❌ Conda install failed"
        fi
    fi
    
    # Manual package installation with specific versions
    echo "   Attempting manual package installation..."
    pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
    pip install transformers==4.35.0
    pip install huggingface_hub==0.19.0
    pip install coremltools==7.1
    pip install numpy==1.24.3
}

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo ""
    echo "📋 Found requirements.txt"
    cat requirements.txt
    
    # Ask user if they want to install
    echo ""
    read -p "Would you like to install these packages? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_packages
    fi
else
    echo ""
    echo "❌ requirements.txt not found"
    echo "💡 Please run this script from the model-converter directory"
fi

# Test imports
echo ""
echo "🧪 Testing Package Imports:"

python3 -c "
import sys
print('   🐍 Python:', sys.version.split()[0])

try:
    import torch
    print('   ✅ PyTorch:', torch.__version__)
except ImportError as e:
    print('   ❌ PyTorch import failed:', e)

try:
    import transformers
    print('   ✅ Transformers:', transformers.__version__)
except ImportError as e:
    print('   ❌ Transformers import failed:', e)

try:
    import huggingface_hub
    print('   ✅ HuggingFace Hub available')
except ImportError as e:
    print('   ❌ HuggingFace Hub import failed:', e)

try:
    import coremltools as ct
    print('   ✅ CoreML Tools:', ct.__version__)
    
    # Test basic functionality
    try:
        import coremltools.converters
        print('   ✅ CoreML converters available')
    except ImportError as e:
        print('   ⚠️  CoreML converters issue:', e)
        
except ImportError as e:
    print('   ❌ CoreML Tools import failed:', e)
    print('   💡 Try: pip uninstall coremltools && pip install coremltools')
"

echo ""
echo "🎯 Setup Complete!"
echo ""
echo "💡 If you encountered issues:"
echo "   1. Try using Python 3.9-3.11 instead of 3.14"
echo "   2. Use a fresh virtual environment"
echo "   3. On Apple Silicon Macs, ensure you're using the correct architecture"
echo "   4. For persistent issues, try conda instead of pip"
echo ""
echo "🚀 To test the converter:"
echo "   python model-converter.py --help"