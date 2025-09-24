from torch.utils.data import Dataset
import pandas as pd

class GCEDataset(Dataset):
    """
    Custom Dataset class for Grammar Correction.
    Loads data from a CSV file and provides access to input and target sentences.
    """
    def __init__(self, path):
        dataset = pd.read_csv(path)
        self.X = dataset['modified'].to_list() # Input sentences (with errors)
        self.y = dataset['sentence'].to_list() # Target sentences (corrected)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
    



def set_trainable_layers(model : callable, n_decoder_unfreeze : int, n_encoder_unfreeze : int):
    """
    Sets the trainable layers of the model by freezing all parameters and unfreezing the last n layers of the decoder and encoder.

    Args:
        model (AutoModelForSeq2SeqLM): The model to modify the parameters of.
        n_decoder_unfreeze (int): Number of decoder layers to unfreeze.
        n_encoder_unfreeze (int): Number of encoder layers to unfreeze.
    """
    # Freeze all model parameters
    for parameters in model.parameters():
        parameters.requires_grad = False

    # Unfreeze last n_decoder_unfreeze layers of the decoder
    n_decoder_layers = len(model.decoder.block)
    for layer in model.decoder.block[n_decoder_layers - n_decoder_unfreeze:]:
        for parameters in layer.parameters():
            parameters.requires_grad = True

    n_encoder_layers = len(model.encoder.block)

    for layer in model.encoder.block[n_encoder_layers- n_encoder_unfreeze:]:
        for parameters in layer.parameters():
            parameters.requires_grad=True
