import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Remains to addapt the neural network for categorical data, because now it is working for continious data
# P.S: Tried one hot encoding but, it didn't worked

class Dataset:
    def __init__(self, path, batch_size):
        # Read from file
        self._data_from_csv = pd.read_csv(path)

        # Transform data to pytorch tensors
        (self._input_matrix, self._target_matrix) = self.transform_data_to_tensor_dataset()
        # input_matrix = input_matrix.reshape(-1, 1)
        self._train_ds = self.create_tensor_dataset(self._input_matrix, self._target_matrix)
        self._data_loader = self.create_data_loader(batch_size)

    # Separate x and y or with other words, target which we want to predict with the features
    def separate_features_target(self):
        df_features = self._data_from_csv.loc[:, self._data_from_csv.columns != 'target']
        df_target = self._data_from_csv['target']
        return df_features, df_target

    # Transform dataframe to the pytorch tensors
    def transform_data_to_tensor_dataset(self):
        (df_features, df_target) = self.separate_features_target()
        features_np = df_features.to_numpy(dtype=np.float32)
        target_np = df_target.to_numpy(dtype=np.float32)
        target_np = target_np.reshape(-1, 1)
        input_matrix = torch.from_numpy(features_np)
        target_matrix = torch.from_numpy(target_np)
        return input_matrix, target_matrix

    # Create Tensor Dataset - structure from pytorch. Have the possibility to access both features on input and target
    def create_tensor_dataset(self, input_matrix, target_matrix):
        return TensorDataset(input_matrix, target_matrix)

    # Create Tensor Loader - structure from pytorch and separate the data in batches. Also shuffle the data.
    def create_data_loader(self, batch_size):
        return DataLoader(self._train_ds, batch_size, shuffle=True)

    def initialize_the_model(self):
        self._model = nn.Linear(self.get_number_of_columns_training(), self.get_number_of_rows_training())
        # Define loss function
        self._loss_function = F.mse_loss
        # Define optimizer
        self._opt = torch.optim.SGD(self._model.parameters(), lr=1e-5)

    def get_number_of_columns_training(self):
        # return len(self._data_from_csv.columns) - 1
        return self._data_from_csv.shape[1] - 1

    def get_number_of_rows_training(self):
        return self._data_from_csv.shape[0]

    def predict(self):
        print(self._model(self._input_matrix))

    # Train model with given number of epochs
    def train_model(self, num_epochs):
            # Repeat for given number of epochs
            for epoch in range(num_epochs):

                # Train with batches of data
                for xb, yb in self._data_loader:
                    # 1. Generate predictions
                    pred = self._model(xb)
                    # yb = torch.reshape(yb, (5, 303))
                    # 2. Calculate loss
                    loss = self._loss_function(pred, yb)

                    # 3. Compute gradients
                    loss.backward()

                    # 4. Update parameters using gradients
                    self._opt.step()

                    # 5. Reset the gradients to zero
                    self._opt.zero_grad()

                # Print the progress
                if (epoch + 1) % 10 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
