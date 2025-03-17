import json
import torch

class MemoryStore:
    def __init__(self, file_path="memory_store.json"):
        self.file_path = file_path
        self.memory = self.load_memory()
    
    def load_memory(self):
        try:
            with open(self.file_path, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_memory(self):
        with open(self.file_path, "w") as file:
            json.dump(self.memory, file, indent=4)
    
    def store(self, key, tensor_data):
        self.memory[key] = tensor_data.tolist()  # Convert tensor to list before storing
        self.save_memory()
    
    def retrieve(self, key):
        if key in self.memory:
            return torch.tensor(self.memory[key])  # Convert back to tensor
        return None

# Example Usage
if __name__ == "__main__":
    mem_store = MemoryStore()
    sample_tensor = torch.randn(32)  # Example latent vector
    mem_store.store("sample_key", sample_tensor)
    retrieved_tensor = mem_store.retrieve("sample_key")
    print("Retrieved Tensor:", retrieved_tensor)
