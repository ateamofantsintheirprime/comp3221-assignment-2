import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy

class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size):
        self.x_train, self.x_test, self.y_train, self.y_test, self.train_samples, self.test_samples = self.get_data(client_id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]

        self.trainloader = DataLoader(self.train_data, batch_size = batch_size)
        self.testloader = DataLoader(self.test_data, batch_size = self.test_samples)

        # Define the Mean Square Error Loss
        self.loss = nn.MSELoss()

        self.model = copy.deepcopy(model)

        self.client_id = client_id

        # Define the Gradient Descent optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
    def get_data(client):
        train_MedInc = []
        train_HouseAge = []
        train_AveRooms = []
        train_AveBedrms = []
        train_Population = []
        train_AveOccup = []
        train_Latitude = []
        train_Longitude = []
        train_MedHouseVal = []

        test_MedInc = []
        test_HouseAge = []
        test_AveRooms = []
        test_AveBedrms = []
        test_Population = []
        test_AveOccup = []
        test_Latitude = []
        test_Longitude = []
        test_MedHouseVal = []

        filename = "FLData/calhousing_train_" + client + ".csv"
        with open(filename, "r") as file:
            #ignores the first line of names of dataset
            next(file)
            for line in file:
                split_data = line.split(",")
                train_MedInc.append(float(split_data[0]))
                train_HouseAge.append(float(split_data[1]))
                train_AveRooms.append(float(split_data[2]))
                train_AveBedrms.append(float(split_data[3]))
                train_Population.append(float(split_data[4]))
                train_AveOccup.append(float(split_data[5]))
                train_Latitude.append(float(split_data[6]))
                train_Longitude.append(float(split_data[7]))
                train_MedHouseVal.append(float(split_data[8]))

        filename = "FLData/calhousing_test_" + client + ".csv"
        with open(filename, "r") as file:
            #ignores the first line of names of dataset
            next(file)
            for line in file:
                split_data = line.split(",")
                test_MedInc.append(float(split_data[0]))
                test_HouseAge.append(float(split_data[1]))
                test_AveRooms.append(float(split_data[2]))
                test_AveBedrms.append(float(split_data[3]))
                test_Population.append(float(split_data[4]))
                test_AveOccup.append(float(split_data[5]))
                test_Latitude.append(float(split_data[6]))
                test_Longitude.append(float(split_data[7]))
                test_MedHouseVal.append(float(split_data[8]))

        x_train_np = np.column_stack((train_MedInc, train_HouseAge, train_AveRooms, train_AveBedrms, train_Population, train_AveOccup, train_Latitude, train_Longitude))
        x_test_np = np.column_stack((test_MedInc, test_HouseAge, test_AveRooms, test_AveBedrms, test_Population, test_AveOccup, test_Latitude, test_Longitude))
        
        x_train = torch.Tensor(x_train_np).view(-1,8).type(torch.float32)
        x_test = torch.Tensor(x_test_np).view(-1,8).type(torch.float32)
        
        y_train = torch.Tensor(train_MedHouseVal).type(torch.float32)
        y_test = torch.Tensor(test_MedHouseVal).type(torch.float32)

        return x_train, x_test, y_train, y_test, len(train_MedHouseVal), len(test_MedHouseVal)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        mse = 0
        for x, y in self.testloader:
            y_pred = self.model(x)
            # Calculate evaluation metrics
            mse += self.loss(y_pred, y)
            print(str(self.id) + ", MSE of client ",self.id, " is: ", mse)
        return mse
    