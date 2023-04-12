import torch

# Create a tensor of shape (5, 3) with some sample weights
weights = torch.tensor([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2],
                        [1.3, 1.4, 1.5]])

# Define a list of indices to keep
indices = [1, 2, 3, 4]

# Create a Boolean mask indicating which indices are in the list
mask = torch.isin(torch.arange(weights.shape[0]), indices)

# Set the weights of indices not in the list to 0
weights[~mask, :] = 0

# Print the result
print(weights)
