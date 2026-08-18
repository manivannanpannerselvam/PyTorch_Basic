"""
Topic 01: Tensors
=================
A tensor is PyTorch's core data structure — a multi-dimensional array,
similar to a NumPy array, but it can also live on a GPU and track
gradients (we'll cover that in Topic 03: Autograd).

Run this file top to bottom, section by section. Try changing values
and re-running to see how the output changes.
"""

import torch

# print("PyTorch version:", torch.__version__)
# print()

# # -----------------------------------------------------------------
# # 1. Creating tensors
# # -----------------------------------------------------------------
# print("--- 1. Creating tensors ---")

# # From a Python list
# t1 = torch.tensor([1, 2, 3])
# print("From list:", t1)

# From a nested list -> 2D tensor (matrix)
# t2 = torch.tensor([[1, 2], [3, 4]])
# print("From nested list (2D):\n", t2)

# # Filled with zeros / ones
# zeros = torch.zeros(1, 3)   # shape (2, 3)
# ones = torch.ones(1, 3)
# print("Zeros:\n", zeros)
# print("Ones:\n", ones)

# # Random values (uniform [0, 1))
# rand = torch.rand(2, 3)
# print("Random:\n", rand)

# # Like NumPy's arange / linspace
# r = torch.arange(0, 10, 2)      # start, end (exclusive), step
# lin = torch.linspace(0, 1, 5)   # 5 evenly spaced points from 0 to 1
# print("arange:", r)
# print("linspace:", lin)

# print()

# # -----------------------------------------------------------------
# # 2. Shape, dtype, device — the 3 things to always check on a tensor
# # -----------------------------------------------------------------
# print("--- 2. Shape, dtype, device ---")

# x = torch.tensor([[1.0, 2.3, 3.0], [4.0, 5.0, 6.0]])
# print("Tensor:\n", x)
# print("shape:", x.shape)        # torch.Size([2, 3]) -> 2 rows, 3 columns
# print("dtype:", x.dtype)        # data type, e.g. torch.float32
# print("device:", x.device)      # where it lives: cpu, cuda, or mps

# # You can force a dtype
# int_tensor = torch.tensor([1.7, 2, 3], dtype=torch.int64)
# print("int dtype:", int_tensor.dtype)

# print()

# # -----------------------------------------------------------------
# # 3. Indexing and slicing — works just like NumPy
# # -----------------------------------------------------------------
# print("--- 3. Indexing and slicing ---")

m = torch.arange(1, 17).reshape(4, 4)  # 3x4 matrix, values 1..12
print("Matrix:\n", m)

# print("First row:", m[0])
# print("Last column:", m[:, -1])
# print("Sub-block (rows 0-1, cols 1-2):\n", m[0:2, 1:3])
# print("Single element (row 1, col 2):", m[1, 2].item())  # .item() -> plain Python number

# print()

# # -----------------------------------------------------------------
# # 4. Reshaping — changing the "view" of the data without changing values
# # -----------------------------------------------------------------
# print("--- 4. Reshaping ---")

# v = torch.arange(6)
# print("Original (shape 6,):", v)

# reshaped = v.reshape(2, 3)
# print("Reshaped to (2, 3):\n", reshaped)

# flattened = reshaped.flatten()
# print("Flattened back:", flattened)

# # -1 means "figure out this dimension automatically"
# auto = v.reshape(3, -1)
# print("Reshape with -1 -> (3, -1):\n", auto)

# print()

# # -----------------------------------------------------------------
# # 5. NumPy interop
# # -----------------------------------------------------------------
# print("--- 5. NumPy interop ---")

# import numpy as np

# np_array = np.array([1, 2, 3])
# from_numpy = torch.from_numpy(np_array)
# print("Tensor from NumPy:", from_numpy)

# back_to_numpy = from_numpy.numpy()
# print("Back to NumPy:", back_to_numpy, type(back_to_numpy))

# print()
# print("Done. Try editing values above and re-running to build intuition.")
