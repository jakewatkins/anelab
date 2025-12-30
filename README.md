# ANE Lab

Projects to learn how to work with Apple's Core ML library so I can access the Neural Engine (ANE) capabilities of Apple Silicon when running models.

# Holding off

I've spent 2 days messing around with Core ML and have figured out that it doesnt do what I need.  I'm looking to build LLM host to build tools and agentic systems.  Core ML doesn't really appear to do that.  I think I could build on top of it to create stuff that would eventually allow me to build the stuff I want, but for now MLX already does most of what I want.  The missing part is that MLX doesnt let you use the Nueral Engine component of the Apple Silicon CPU.  It just gives you access to the CPU and GPU.  
Part of the driver behind my decision came from trying to implement a model converter.  In order to run a model on ANE it first has to be converted to its format.  model-converter was supposed to let you give it a HuggingFace model name, it would download it and convert it.  Then the host I'd eventually build would compile it and use it.  Unfortunately, most models I want to use (IBM's Granite models for example) can't be converted.  Or at least my attempts failed.
Instead of fighting this, I'm going to focus my efforts on using MLX and build there.  Perhaps CoreML will evolve and support what I want, but for now I'll walk a different path.