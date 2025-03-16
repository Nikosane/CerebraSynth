class ModelTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def train(self, data_loader, epochs=10):
        for epoch in range(epochs):
            epoch_loss = 0.0
            for data in data_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader):.4f}")
