import torch

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved
print("Device name:", torch.cuda.get_device_name(0))
print("Total memory:", t)
print("Reserved memory:", r)
print("Allocated memory:", a)
print("Free memory:", f)

