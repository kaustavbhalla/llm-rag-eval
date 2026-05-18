import torch
import time

print("=== PyTorch WSL 2 GPU Test ===")

# 1. Check if PyTorch can see CUDA
if torch.cuda.is_available():
    print("✅ CUDA is available!")
    
    # 2. Get device properties
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Device Name: {gpu_name}")
    print(f"✅ Number of GPUs: {torch.cuda.device_count()}")
    
    # 3. Perform a stress test calculation
    print("\nAllocating tensors and running matrix multiplication on the GPU...")
    
    # Create large random matrices directly on the GPU
    size = 10000 
    
    try:
        start_time = time.time()
        
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)
        
        # Heavy math operation
        c = torch.matmul(a, b)
        
        # Wait for the GPU computation to finish before stopping the clock
        torch.cuda.synchronize() 
        
        end_time = time.time()
        
        print(f"✅ Success! Matrix math completed in {end_time - start_time:.4f} seconds.")
        print("Your WSL 2 environment is fully accelerated and ready for deep learning.")
        
    except Exception as e:
        print(f"❌ An error occurred during computation: {e}")
        
else:
    print("❌ CUDA is NOT available. PyTorch is defaulting to the CPU.")
    print("Please verify the CUDA toolkit installation steps.")
