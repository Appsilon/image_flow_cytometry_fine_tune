import torch

def print_cuda_memory():
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # Convert to MB
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # Convert to MB
        free_memory = total_memory - reserved

        print(f"Device {i}:")
        print(f"  Allocated Memory: {allocated:.2f} MB")
        print(f"  Reserved Memory: {reserved:.2f} MB")
        print(f"  Free Memory: {free_memory:.2f} MB")
        print(f"  Total Memory: {total_memory:.2f} MB")