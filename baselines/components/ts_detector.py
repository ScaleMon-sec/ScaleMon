import torch
import numpy as np

class TsDetector:
    
    def __init__(self, model, device, batch_size):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def __call__(self, chunked_sequence):
        errors = []

        with torch.no_grad():
            for i in range(0, len(chunked_sequence), self.batch_size):
                batch = chunked_sequence[i:i + self.batch_size]

                if isinstance(batch, np.ndarray):
                    batch = torch.from_numpy(batch)

                batch = batch.to(self.device)

                mse = self.compute_reconstruction_error(batch)
                errors.append(mse)
                
        errors = torch.cat(errors, dim=0)

        return errors.max().item()

    def compute_reconstruction_error(self, x):
        x_hat = self.model(x)

        mse = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        return mse.cpu()
