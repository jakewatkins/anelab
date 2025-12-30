#!/usr/bin/env python3
"""
Model Converter - HuggingFace to Core ML Converter

A command-line tool that downloads models from HuggingFace Hub and converts 
them to Apple's Core ML format for use in iOS, macOS, and other Apple platforms.

Usage:
    python model-converter.py <huggingface_model_name>

Example:
    python model-converter.py mlx-community/mistralai_Ministral-3-14B-Instruct-2512-MLX-MXFP4
"""

import argparse
import logging
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from huggingface_hub import hf_hub_download, repo_info, HfApi
    
    # Check for CoreML availability and provide helpful error messages
    try:
        import coremltools as ct
        print(f"✅ CoreML Tools version: {ct.__version__}")
        print(f"✅ Torch version: {torch.__version__}")
        
        # Check if we're on a compatible platform
        import platform
        if platform.system() != 'Darwin':
            print("⚠️  Warning: Core ML works best on macOS. Other platforms may have limited functionality.")
            
    except ImportError as coreml_error:
        print(f"❌ CoreML Tools import failed: {coreml_error}")
        print("This might be due to:")
        print("1. Incompatible Python version (3.14 may not be supported)")
        print("2. Missing native libraries")
        print("3. Architecture mismatch (Intel vs Apple Silicon)")
        print("\n💡 Suggested fixes:")
        print("- Try Python 3.9-3.11 (more stable with CoreML)")
        print("- Reinstall coremltools: pip uninstall coremltools && pip install coremltools")
        print("- Use conda instead: conda install -c apple coremltools")
        raise coreml_error
        
except ImportError as e:
    print(f"Error: Required dependency not found: {e}")
    print("Please install required dependencies:")
    print("pip install torch transformers huggingface_hub coremltools")
    sys.exit(1)


class ModelConverter:
    """Main class for converting HuggingFace models to Core ML format."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.logger = self._setup_logging()
        self.temp_dir = None
        self.hf_api = HfApi()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def validate_model_name(self) -> bool:
        """
        Validate the HuggingFace model name format and availability.
        
        Returns:
            bool: True if valid and accessible, False otherwise
        """
        try:
            # Some popular models don't follow user/model-name format (e.g., gpt2, bert-base-uncased)
            # So we'll check availability directly rather than strict format checking
            
            self.logger.info(f"Validating model: {self.model_name}")
            info = repo_info(self.model_name)
            self.logger.info(f"Model found: {info.modelId}")
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            if "not found" in str(e).lower():
                self.logger.error("This model doesn't exist on HuggingFace Hub")
                self.logger.error("💡 Use --suggest to see available models")
            elif "format" in str(e).lower():
                self.logger.error("Model name format issue")
                self.logger.error("💡 Expected format: 'user/model-name' or just 'model-name' for popular models")
            return False
    
    def download_model(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Download the model and tokenizer from HuggingFace Hub.
        
        Returns:
            Tuple[Optional[str], Optional[str]]: Paths to model and tokenizer, or None if failed
        """
        try:
            self.logger.info(f"Downloading model: {self.model_name}")
            
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="model_converter_")
            
            # Download model files
            model_path = os.path.join(self.temp_dir, "model")
            tokenizer_path = os.path.join(self.temp_dir, "tokenizer")
            
            # Try to download tokenizer first (lighter operation)
            try:
                self.logger.info("Downloading tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=tokenizer_path,
                    trust_remote_code=False  # More secure default
                )
            except Exception as e:
                self.logger.error(f"Tokenizer download failed: {str(e)}")
                if "trust_remote_code" in str(e):
                    self.logger.info("Trying with trust_remote_code=True...")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            cache_dir=tokenizer_path,
                            trust_remote_code=True
                        )
                    except Exception as e2:
                        self.logger.error(f"Tokenizer download failed even with trust_remote_code: {str(e2)}")
                        return None, None
                else:
                    return None, None
            
            # Try different model loading strategies
            model = None
            model_type = "unknown"
            
            # Strategy 1: Try AutoModelForCausalLM
            try:
                self.logger.info("Attempting to load as CausalLM...")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=False,
                    cache_dir=model_path
                )
                model_type = "causal_lm"
            except Exception as e1:
                self.logger.info(f"CausalLM failed: {str(e1)}")
                
                # Strategy 2: Try with trust_remote_code
                try:
                    self.logger.info("Trying CausalLM with trust_remote_code=True...")
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        cache_dir=model_path
                    )
                    model_type = "causal_lm_trusted"
                except Exception as e2:
                    self.logger.info(f"CausalLM with trust_remote_code failed: {str(e2)}")
                    
                    # Strategy 3: Try base AutoModel
                    try:
                        self.logger.info("Trying base AutoModel...")
                        model = AutoModel.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float32,
                            trust_remote_code=False,
                            cache_dir=model_path
                        )
                        model_type = "base_model"
                    except Exception as e3:
                        self.logger.info(f"Base model failed: {str(e3)}")
                        
                        # Strategy 4: Last resort - base model with trust_remote_code
                        try:
                            self.logger.info("Last attempt: base model with trust_remote_code=True...")
                            model = AutoModel.from_pretrained(
                                self.model_name,
                                torch_dtype=torch.float32,
                                trust_remote_code=True,
                                cache_dir=model_path
                            )
                            model_type = "base_model_trusted"
                        except Exception as e4:
                            self.logger.error(f"All model loading strategies failed.")
                            self.logger.error(f"Final error: {str(e4)}")
                            
                            # Provide specific guidance based on error types
                            error_msg = str(e4).lower()
                            if "granitemoehybrid" in error_msg:
                                self.logger.error("❌ This model uses an unsupported architecture (Granite MoE Hybrid)")
                                self.logger.error("💡 Try a standard transformer model instead")
                            elif "rope_scaling" in error_msg:
                                self.logger.error("❌ This model has incompatible RoPE scaling configuration")
                                self.logger.error("💡 Try a different version of this model or a simpler model")
                            elif "mlx" in self.model_name.lower():
                                self.logger.error("❌ MLX models are optimized for Apple Silicon and may not be compatible")
                                self.logger.error("💡 Try the original (non-MLX) version of this model")
                            elif "4bit" in self.model_name.lower() or "8bit" in self.model_name.lower():
                                self.logger.error("❌ Quantized models may not be compatible with Core ML conversion")
                                self.logger.error("💡 Try the full-precision version of this model")
                            
                            return None, None
            
            if model is None:
                self.logger.error("Failed to load model with any strategy")
                return None, None
            
            self.logger.info(f"✅ Successfully downloaded {model_type}")
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Unexpected error during model download: {str(e)}")
            return None, None
    
    def convert_to_coreml(self, model, tokenizer) -> Optional[str]:
        """
        Convert the downloaded model to Core ML format.
        
        Args:
            model: The HuggingFace model
            tokenizer: The HuggingFace tokenizer
            
        Returns:
            Optional[str]: Path to converted .mlpackage file, or None if failed
        """
        try:
            self.logger.info("Converting model to Core ML format...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Get model configuration
            config = model.config
            
            # Check if model is too large for conversion
            num_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Model has {num_params:,} parameters")
            
            if num_params > 1_000_000_000:  # 1B parameters
                self.logger.warning("Large model detected. This may cause memory issues during conversion.")
                self.logger.warning("Consider using a smaller model or a machine with more RAM.")
            
            # Create a sample input for tracing with proper attention mask
            vocab_size = getattr(config, 'vocab_size', 32000)
            max_seq_len = min(getattr(config, 'max_position_embeddings', 512), 32)  # Even smaller for compatibility
            
            sample_input_ids = torch.randint(
                low=1,  # Avoid padding token (usually 0)
                high=min(vocab_size, 32000),  # Limit vocab size for safety
                size=(1, max_seq_len),
                dtype=torch.long
            )
            
            self.logger.info(f"Using input shape: {sample_input_ids.shape}")
            
            # Try different conversion approaches
            try:
                self.logger.info("Attempting direct model conversion (no tracing)...")
                
                # Create a simple wrapper that only takes input_ids
                class SimpleModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                        # Check what type of model we have
                        self.is_causal_lm = hasattr(model, 'lm_head') or 'GPT' in str(type(model)) or 'Causal' in str(type(model))
                        self.is_bert_style = 'Bert' in str(type(model)) or 'DistilBert' in str(type(model))
                        
                    def forward(self, input_ids):
                        try:
                            if self.is_causal_lm:
                                # For causal LM models (GPT-style)
                                outputs = self.model(input_ids=input_ids, use_cache=False, return_dict=False)
                                return outputs[0] if isinstance(outputs, tuple) else outputs.logits
                            elif self.is_bert_style:
                                # For BERT-style models (encoder only)
                                outputs = self.model(input_ids=input_ids, return_dict=False)
                                # Return the last hidden state
                                return outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
                            else:
                                # Generic fallback
                                outputs = self.model(input_ids=input_ids, return_dict=False)
                                return outputs[0] if isinstance(outputs, tuple) else outputs
                        except Exception as e:
                            # Ultimate fallback - just pass input_ids
                            outputs = self.model(input_ids)
                            return outputs[0] if isinstance(outputs, tuple) else outputs
                
                wrapped_model = SimpleModelWrapper(model)
                wrapped_model.eval()
                
                # Test the wrapper first
                with torch.no_grad():
                    test_output = wrapped_model(sample_input_ids)
                    self.logger.info(f"Model wrapper test successful. Output shape: {test_output.shape}")
                
                # Try scripting instead of tracing
                try:
                    self.logger.info("Attempting torch.jit.script...")
                    scripted_model = torch.jit.script(wrapped_model)
                    use_scripted = True
                except Exception as script_error:
                    self.logger.info(f"Scripting failed: {str(script_error)}")
                    self.logger.info("Attempting torch.jit.trace...")
                    
                    try:
                        with torch.no_grad():
                            traced_model = torch.jit.trace(wrapped_model, sample_input_ids)
                        use_scripted = False
                        scripted_model = traced_model
                    except Exception as trace_error:
                        self.logger.error(f"Both scripting and tracing failed!")
                        self.logger.error(f"Script error: {str(script_error)}")
                        self.logger.error(f"Trace error: {str(trace_error)}")
                        return None
                
                # Convert to Core ML with multiple fallback options
                self.logger.info("Converting to Core ML format...")
                
                # First attempt: Standard conversion
                try:
                    coreml_model = ct.convert(
                        scripted_model,
                        inputs=[ct.TensorType(
                            name="input_ids",
                            shape=sample_input_ids.shape,
                            dtype=np.int64  # Use numpy dtype
                        )],
                        minimum_deployment_target=ct.target.iOS15,  # More compatible target
                        compute_precision=ct.precision.FLOAT16,
                        compute_units=ct.ComputeUnit.CPU_ONLY
                    )
                    self.logger.info("✅ Standard Core ML conversion successful")
                    
                except Exception as convert_error1:
                    self.logger.info(f"Standard conversion failed: {str(convert_error1)}")
                    
                    # Fallback attempt: Different settings
                    try:
                        self.logger.info("Trying fallback conversion settings...")
                        coreml_model = ct.convert(
                            scripted_model,
                            inputs=[ct.TensorType(
                                name="input_ids", 
                                shape=ct.Shape(shape=sample_input_ids.shape),  # Explicit shape
                                dtype=np.int64  # Use numpy dtype
                            )],
                            minimum_deployment_target=ct.target.iOS13,
                            compute_precision=ct.precision.FLOAT32,  # Higher precision
                            compute_units=ct.ComputeUnit.CPU_ONLY
                        )
                        self.logger.info("✅ Fallback Core ML conversion successful")
                        
                    except Exception as convert_error2:
                        self.logger.error(f"All conversion attempts failed!")
                        self.logger.error(f"Error 1: {str(convert_error1)}")
                        self.logger.error(f"Error 2: {str(convert_error2)}")
                        self.logger.error("This model architecture may not be compatible with Core ML")
                        return None
                
            except Exception as wrapper_error:
                self.logger.error(f"Model wrapper creation failed: {str(wrapper_error)}")
                return None
            
            # Generate output filename
            model_filename = self.model_name.replace('/', '_').replace('-', '_') + '.mlpackage'
            output_path = os.path.join(os.getcwd(), model_filename)
            
            # Add metadata before saving
            coreml_model.short_description = f"Core ML model converted from {self.model_name}"
            coreml_model.author = "Model Converter Tool"
            coreml_model.license = "See original model license on HuggingFace Hub"
            coreml_model.version = "1.0"
            
            # Save the model
            self.logger.info(f"Saving Core ML model to: {output_path}")
            coreml_model.save(output_path)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Core ML conversion failed: {str(e)}")
            
            # Provide specific guidance based on error type
            if "cache" in str(e).lower():
                self.logger.error("💡 Model uses dynamic caching which is not compatible with static conversion")
            elif "memory" in str(e).lower():
                self.logger.error("💡 Try using a smaller model or increase available RAM")
            elif "trace" in str(e).lower():
                self.logger.error("💡 This model architecture may not support torch.jit.trace or script")
            elif "coreml" in str(e).lower():
                self.logger.error("💡 Check CoreML Tools compatibility with your model architecture")
            
            return None
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up temporary files")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temp directory: {str(e)}")
    
    def convert(self) -> bool:
        """
        Main conversion workflow.
        
        Returns:
            bool: True if conversion succeeded, False otherwise
        """
        try:
            # Validate model
            if not self.validate_model_name():
                return False
            
            # Download model
            model, tokenizer = self.download_model()
            if model is None or tokenizer is None:
                return False
            
            # Convert to Core ML
            output_path = self.convert_to_coreml(model, tokenizer)
            if output_path is None:
                return False
            
            self.logger.info(f"✅ Conversion completed successfully!")
            self.logger.info(f"📁 Core ML model saved to: {output_path}")
            
            # Display file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            self.logger.info(f"📊 Model size: {file_size:.1f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            return False
        
        finally:
            self.cleanup()


def suggest_compatible_models():
    """Print a list of known compatible models for testing."""
    print("\n🎯 Suggested Compatible Models (known to work well):")
    print("=" * 60)
    print("📝 Text Generation (Small):")
    print("   • distilgpt2")
    print("   • gpt2")
    print("   • microsoft/DialoGPT-small")
    print("")
    print("🤖 BERT-style Models:")
    print("   • distilbert-base-uncased")
    print("   • bert-base-uncased")
    print("   • microsoft/deberta-v3-small")
    print("")
    print("💬 Conversational Models:")
    print("   • microsoft/DialoGPT-medium")
    print("   • facebook/blenderbot_small-90M")
    print("")
    print("⚠️  Models to AVOID:")
    print("   ❌ MLX-optimized models (Apple Silicon specific)")
    print("   ❌ Quantized models (4bit, 8bit)")
    print("   ❌ Very large models (>7B parameters)")
    print("   ❌ Models with custom architectures (MoE, hybrid)")
    print("")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Apple Core ML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model-converter.py mlx-community/mistralai_Ministral-3-14B-Instruct-2512-MLX-MXFP4
  python model-converter.py microsoft/DialoGPT-medium
  python model-converter.py distilbert-base-uncased
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",  # Make model_name optional
        help="HuggingFace model name (format: user/model-name or organization/model-name)"
    )
    
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Show list of compatible models and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Handle --suggest flag
    if args.suggest:
        suggest_compatible_models()
        sys.exit(0)
    
    # Validate that model_name was provided
    if not hasattr(args, 'model_name') or not args.model_name:
        parser.print_help()
        print("\n❌ Error: model_name is required")
        print("💡 Use --suggest to see compatible models")
        sys.exit(1)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter and run conversion
    converter = ModelConverter(args.model_name)
    
    print(f"🚀 Model Converter v1.0")
    print(f"📥 Converting: {args.model_name}")
    print(f"💾 Output directory: {os.getcwd()}")
    print("-" * 50)
    
    success = converter.convert()
    
    if success:
        print("\n✅ Conversion completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Conversion failed!")
        print("💡 Try these solutions:")
        print("   1. Use --suggest to see compatible models")
        print("   2. Avoid MLX, quantized, or very large models")
        print("   3. Check if the model architecture is supported")
        print("   4. Try a simpler/smaller model first")
        sys.exit(1)


if __name__ == "__main__":
    main()