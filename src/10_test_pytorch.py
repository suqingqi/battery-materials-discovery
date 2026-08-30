import torch

from torch_geometric.data import Data


print("PyTorch version:")
print(torch.__version__)


print()
print("MPS available:")
print(torch.backends.mps.is_available())


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


print()
print("Selected device:")
print(device)


x = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)


edge_index = torch.tensor(
    [
        [0, 1, 1, 2],
        [1, 0, 2, 1],
    ],
    dtype=torch.long,
)


data = Data(
    x=x,
    edge_index=edge_index,
)


print()
print("Graph:")
print(data)


data = data.to(device)


print()
print("Graph device:")
print(data.x.device)


print()
print("STEP 10 test completed.")