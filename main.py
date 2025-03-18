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
