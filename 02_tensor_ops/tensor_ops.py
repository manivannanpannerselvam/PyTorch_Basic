"""
Topic 02: Tensor operations
============================
Math on tensors, broadcasting, matrix multiply, and moving tensors
between devices (CPU / GPU / MPS).

Run this file top to bottom, section by section. Uncomment one block
at a time, run it, read the output, then move to the next.
"""

import torch

print("PyTorch version:", torch.__version__)
print()

# -----------------------------------------------------------------
# 1. Element-wise math
# -----------------------------------------------------------------
# print("--- 1. Element-wise math ---")

# a = torch.tensor([1.0, 2.0, 3.0])
# b = torch.tensor([4.0, 5.0, 6.0])

# print("a + b:", a + b)
# print("a - b:", a - b)
# print("a * b:", a * b)          # element-wise multiply, NOT matrix multiply
# print("a / b:", a / b)
# print("a ** 2:", a ** 2)

# Same ops also exist as functions, useful when chaining
# print("torch.add(a, b):", torch.add(a, b))

# print()

# # -----------------------------------------------------------------
# 2. Reductions — collapsing a tensor down to fewer values
# -----------------------------------------------------------------
# print("--- 2. Reductions ---")

# m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print("Matrix:\n", m)

# print("sum (all):", m.sum())
# print("sum (dim=0, down columns):", m.sum(dim=0))
# print("sum (dim=1, across rows):", m.sum(dim=1))
# print("mean:", m.mean())
# print("max:", m.max())
# print("argmax:", m.argmax())    # index of the max value in the flattened tensor

# print()

# -----------------------------------------------------------------
# 3. Broadcasting — operating on tensors of different shapes
# -----------------------------------------------------------------
# print("--- 3. Broadcasting ---")

# A smaller tensor is "stretched" to match a larger one, without
# copying data, when their shapes are compatible from the right.
m = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # shape (2, 3)
row = torch.tensor([10.0, 20.0, 30.0])                  # shape (3,)
scalar = 100.0                                          # shape ()

# print("Matrix + row (broadcast across rows):\n", m + row)
# print("Matrix + scalar (broadcast to every element):\n", m + scalar)

# Shapes must be compatible: equal, or one of them is 1
col = torch.tensor([[1.0], [2.0]])   # shape (2, 1)
print("Matrix + col (broadcast across columns):\n", m + col)

# print()

# -----------------------------------------------------------------
# 4. Matrix multiply — different from element-wise multiply
# -----------------------------------------------------------------
# print("--- 4. Matrix multiply ---")

# x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # shape (2, 2)
# y = torch.tensor([[5.0, 6.0], [7.0, 8.0]])   # shape (2, 2)

# print("x * y (element-wise):\n", x * y)
# print("x @ y (matrix multiply):\n", x @ y)
# print("torch.matmul(x, y):\n", torch.matmul(x, y))

# print()

# -----------------------------------------------------------------
# 5. Device movement — CPU, GPU (cuda), Apple Silicon (mps)
# -----------------------------------------------------------------
# print("--- 5. Device movement ---")

# t = torch.tensor([1.0, 2.0, 3.0])
# print("Starts on:", t.device)

# # Check what's available on this machine
# print("CUDA available:", torch.cuda.is_available())
# print("MPS available:", torch.backends.mps.is_available())

# device = "mps" if torch.backends.mps.is_available() else "cpu"
# t_on_device = t.to(device)
# print(f"Moved to {device}:", t_on_device.device)

# # Tensors on different devices can't be combined directly —
# # both operands must be on the same device.

# print()
# print("Done. Try editing values above and re-running to build intuition.")
