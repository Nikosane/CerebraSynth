import torch
from models.autoencoder import Autoencoder
from models.transformer import TransformerAutoencoder
from models.trainer import ModelTrainer
from memory_store import MemoryStore
from utils.logger import logger
from utils.config import Config


def main():
    # Select Model Type
    model_choice = input("Choose model (1: Autoencoder, 2: Transformer): ")

    if model_choice == '1':
        model = Autoencoder(Config.INPUT_DIM, Config.LATENT_DIM)
        print("Using Autoencoder")
    elif model_choice == '2':
        model = TransformerAutoencoder(Config.INPUT_DIM, Config.LATENT_DIM, Config.NUM_HEADS, Config.NUM_LAYERS)
        print("Using Transformer Autoencoder")
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Initialize Trainer and Memory Store
    trainer = ModelTrainer(model, lr=Config.LEARNING_RATE)
    memory_store = MemoryStore()

    # Simulate Data (Placeholder for actual data loader)
    data_loader = [torch.randn(1, Config.INPUT_DIM) for _ in range(100)]

    # Train Model
    print("Starting Training...")
    trainer.train(data_loader, epochs=Config.EPOCHS)

    # Compress and Store Memory
    sample_data = torch.randn(1, Config.INPUT_DIM)
    compressed_data = model.encode(sample_data)
    memory_store.store("sample_compressed", compressed_data)

    print("Training Complete. Compressed Data Stored.")

if __name__ == "__main__":
    main()

