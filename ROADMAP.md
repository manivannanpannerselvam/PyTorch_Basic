# PyTorch Learning Roadmap

We build this **one topic at a time** — code, run, debug, understand — before moving to the next.
Check items off as we complete them. Nothing beyond the current topic is created in advance.

## Basics
- [ ] 01. Tensors — creation, shapes, dtypes, indexing, reshaping
- [ ] 02. Tensor operations — math, broadcasting, matrix multiply, GPU/MPS movement
- [ ] 03. Autograd — how gradients work, `requires_grad`, `.backward()`
- [ ] 04. Linear regression from scratch — manual gradient descent using tensors

## Core Deep Learning
- [ ] 05. `nn.Module` basics — layers, forward pass
- [ ] 06. Loss functions & optimizers
- [ ] 07. Datasets & DataLoaders
- [ ] 08. Full training loop — train/val split, metrics, epochs
- [ ] 09. Activation functions (ReLU, sigmoid, softmax, etc.)

## Intermediate
- [ ] 10. CNNs — convolution, pooling, image classification
- [ ] 11. Regularization — dropout, batch norm, weight decay, early stopping
- [ ] 12. Saving & loading models — checkpoints, `state_dict`
- [ ] 13. Custom datasets & transforms
- [ ] 14. GPU/device management best practices

## Advanced
- [ ] 15. RNNs / LSTMs — sequence data
- [ ] 16. Transfer learning
- [ ] 17. Attention & Transformers basics
- [ ] 18. Custom autograd functions & hooks
- [ ] 19. Model export & deployment (TorchScript / ONNX)
- [ ] 20. Distributed training & performance profiling

## Capstone
- [ ] 21. End-to-end project combining the above

---
## Environment setup (new machine)

Requires Python 3.12 (PyTorch does not yet support very new Python versions like 3.14).

```bash
python3.12 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify it worked:
```bash
python -c "import torch; print(torch.__version__)"
```

Then run any lesson, e.g.:
```bash
python 01_tensors/tensors.py
```

`.venv/` itself is not committed to git (see `.gitignore`) — always recreate it from `requirements.txt` on each machine.
