---
layout: default
title: PyTorch Tutorial for AI Engineering
---

# PyTorch Tutorial for AI Engineering

This short tutorial gives you the PyTorch knowledge needed for the rest of *AI Engineering Interviews*. The goal is not to make you a PyTorch expert. The goal is to make you comfortable reading, writing, and explaining the kind of PyTorch code that appears in AI engineering interviews: tensors, gradients, models, datasets, and training loops.

## What you should know after this tutorial

By the end, you should be able to:

- Create and manipulate tensors.
- Move tensors and models to CPU, GPU, or Apple Silicon/MPS.
- Use automatic differentiation with `loss.backward()`.
- Define a neural network with `torch.nn.Module`.
- Load data with `Dataset` and `DataLoader`.
- Write a standard training loop.
- Save and load model weights.
- Explain what each part of a PyTorch training script is doing.

---

## 1. The mental model

PyTorch is built around a few core ideas:

| Concept | What it means |
|---|---|
| `torch.Tensor` | The main data structure. Similar to a NumPy array, but designed for deep learning and accelerator execution. |
| `autograd` | PyTorch's automatic differentiation system. It tracks operations and computes gradients. |
| `nn.Module` | The base class for neural network models and layers. |
| `Dataset` | Defines how to access one training example. |
| `DataLoader` | Turns a dataset into shuffled mini-batches. |
| `Optimizer` | Updates model parameters using gradients. |

A PyTorch training loop usually follows this pattern:

```python
for inputs, labels in dataloader:
    predictions = model(inputs)
    loss = loss_fn(predictions, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This loop is the heart of supervised training.

---

## 2. Tensors

A tensor is a multi-dimensional array. Scalars, vectors, matrices, and batches of data can all be represented as tensors.

```python
import torch

# Scalar: shape []
x = torch.tensor(3.0)

# Vector: shape [3]
v = torch.tensor([1.0, 2.0, 3.0])

# Matrix: shape [2, 3]
m = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

# Random tensor: shape [4, 5]
r = torch.randn(4, 5)

print(x.shape)
print(v.shape)
print(m.shape)
print(r.shape)
```

Common tensor operations:

```python
a = torch.randn(2, 3)
b = torch.randn(3, 4)

# Matrix multiplication: [2, 3] @ [3, 4] -> [2, 4]
c = a @ b

# Element-wise operations
d = torch.relu(c)
e = torch.softmax(c, dim=-1)

print(c.shape)
print(e.sum(dim=-1))  # each row sums to 1
```

### Shape discipline

Most PyTorch bugs are shape bugs. In AI engineering interviews, you should be able to reason about shapes.

For example, in a language model:

```text
input_ids:          [batch_size, sequence_length]
token_embeddings:  [batch_size, sequence_length, hidden_dim]
logits:            [batch_size, sequence_length, vocab_size]
```

When something breaks, print shapes first:

```python
print("inputs:", inputs.shape)
print("labels:", labels.shape)
print("logits:", logits.shape)
```

---

## 3. Devices: CPU, CUDA, and MPS

PyTorch tensors and models must live on the same device. On a laptop, you may use CPU. On NVIDIA GPUs, use CUDA. On Apple Silicon, you may use MPS.

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
```

Move tensors and models to the device:

```python
x = torch.randn(32, 10).to(device)

model = torch.nn.Linear(10, 1).to(device)
y = model(x)
```

A common error is having the model on one device and the data on another.

---

## 4. Autograd: how PyTorch computes gradients

PyTorch can automatically compute derivatives of a loss with respect to model parameters.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2 + 3 * x + 1
y.backward()

print(x.grad)  # derivative of x^2 + 3x + 1 at x=2 is 2x + 3 = 7
```

What happened?

1. `requires_grad=True` tells PyTorch to track operations involving `x`.
2. `y.backward()` computes the gradient of `y` with respect to `x`.
3. The result is stored in `x.grad`.

In neural network training, the same idea applies, but the loss depends on many parameters instead of one scalar.

---

## 5. Building models with `nn.Module`

In PyTorch, a model is usually a class that inherits from `torch.nn.Module`.

```python
import torch
from torch import nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)
```

Usage:

```python
model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=2)

x = torch.randn(5, 10)      # 5 examples, 10 features each
logits = model(x)           # shape: [5, 2]

print(logits.shape)
```

The `forward` method defines the computation. You do not usually call `model.forward(x)` directly. You call `model(x)`, which lets PyTorch run hooks and internal module logic correctly.

---

## 6. Loss functions

A loss function measures how wrong the model is.

For regression:

```python
loss_fn = nn.MSELoss()
```

For classification with integer labels:

```python
loss_fn = nn.CrossEntropyLoss()
```

Important: `nn.CrossEntropyLoss()` expects raw logits, not probabilities. Do **not** apply `softmax` before passing logits into cross-entropy.

Correct:

```python
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
```

Usually incorrect:

```python
probs = torch.softmax(model(x), dim=-1)
loss = nn.CrossEntropyLoss()(probs, labels)
```

---

## 7. Dataset and DataLoader

A `Dataset` defines how to access one example. A `DataLoader` batches, shuffles, and iterates over the dataset.

Here is a small synthetic classification dataset:

```python
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int = 1000, input_dim: int = 10):
        self.x = torch.randn(n_samples, input_dim)

        # Create labels from a simple rule.
        # If the sum of the first three features is positive, label = 1; else label = 0.
        scores = self.x[:, :3].sum(dim=1)
        self.y = (scores > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

Create a dataloader:

```python
dataset = SyntheticDataset(n_samples=1000, input_dim=10)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
)
```

Inspect one batch:

```python
batch_x, batch_y = next(iter(dataloader))

print(batch_x.shape)  # [32, 10]
print(batch_y.shape)  # [32]
```

---

## 8. A complete training loop

This example trains a small neural network on the synthetic dataset.

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)

class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int = 1000, input_dim: int = 10):
        self.x = torch.randn(n_samples, input_dim)
        scores = self.x[:, :3].sum(dim=1)
        self.y = (scores > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dataset = SyntheticDataset(n_samples=1000, input_dim=10)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=2).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        predictions = logits.argmax(dim=-1)
        correct += (predictions == batch_y).sum().item()
        total += batch_x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
```

What each line means:

| Code | Meaning |
|---|---|
| `model.train()` | Puts the model in training mode. Important for dropout and normalization layers. |
| `logits = model(batch_x)` | Runs the forward pass. |
| `loss = loss_fn(logits, batch_y)` | Computes how wrong the model is. |
| `optimizer.zero_grad()` | Clears old gradients from the previous step. |
| `loss.backward()` | Computes gradients through backpropagation. |
| `optimizer.step()` | Updates model parameters. |
| `loss.item()` | Converts a scalar tensor into a Python number for logging. |

---

## 9. Evaluation mode

During evaluation, you usually do not want PyTorch to track gradients.

```python
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        predictions = logits.argmax(dim=-1)

        correct += (predictions == batch_y).sum().item()
        total += batch_x.size(0)

print("Accuracy:", correct / total)
```

Two important pieces:

```python
model.eval()
```

This switches the model to evaluation mode.

```python
with torch.no_grad():
```

This disables gradient tracking, which saves memory and computation.

---

## 10. Saving and loading models

Save model weights:

```python
torch.save(model.state_dict(), "simple_classifier.pt")
```

Load model weights:

```python
model = SimpleClassifier(input_dim=10, hidden_dim=32, num_classes=2)
model.load_state_dict(torch.load("simple_classifier.pt", map_location="cpu"))
model.eval()
```

In most cases, save the `state_dict`, not the entire model object. The `state_dict` contains the learned weights and is more portable.

---

## 11. Common PyTorch mistakes

### Mistake 1: Forgetting `optimizer.zero_grad()`

Gradients accumulate by default in PyTorch. If you forget to clear them, each update will include gradients from previous batches.

Correct:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Mistake 2: Applying softmax before `CrossEntropyLoss`

`nn.CrossEntropyLoss()` already includes the log-softmax operation internally.

Correct:

```python
loss = nn.CrossEntropyLoss()(logits, labels)
```

### Mistake 3: Device mismatch

If your model is on GPU but your input is on CPU, PyTorch will raise an error.

Correct:

```python
model = model.to(device)
inputs = inputs.to(device)
labels = labels.to(device)
```

### Mistake 4: Wrong label shape or dtype

For classification with `CrossEntropyLoss`, labels should usually be:

```text
shape: [batch_size]
dtype: torch.long
```

not one-hot vectors.

### Mistake 5: Not switching between train and eval mode

Use:

```python
model.train()
```

during training, and:

```python
model.eval()
```

during evaluation or inference.

---

## 12. How this connects to large language models

The same PyTorch ideas scale up to LLMs.

| Small tutorial example | LLM equivalent |
|---|---|
| Input features `[batch, input_dim]` | Token embeddings `[batch, sequence, hidden_dim]` |
| `nn.Linear` layer | Q/K/V projections, MLP layers, output projection |
| Cross-entropy over two classes | Cross-entropy over vocabulary tokens |
| Dataset of synthetic examples | Dataset of tokenized text sequences |
| Training loop | Distributed pretraining or fine-tuning loop |
| `state_dict` | Model checkpoint |

A transformer is larger and more complex than the small classifier above, but the training pattern is the same:

```python
logits = model(input_ids)
loss = loss_fn(logits, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The key difference is scale: LLMs involve billions of parameters, huge datasets, specialized kernels, distributed training, mixed precision, checkpointing, and careful memory management. But the PyTorch foundation remains the same.

---

## 13. Interview checklist

For AI engineering interviews, you should be able to explain:

- What a tensor is and how tensor shapes flow through a model.
- Why we call `optimizer.zero_grad()` before `loss.backward()`.
- What `loss.backward()` computes.
- What `optimizer.step()` changes.
- Why `CrossEntropyLoss` expects logits.
- Why `model.train()` and `model.eval()` matter.
- How `Dataset` and `DataLoader` work.
- How to move tensors and models to GPU.
- How this small training loop maps to LLM fine-tuning.

If you can explain these clearly, you have enough PyTorch foundation to follow most AI engineering code examples.
