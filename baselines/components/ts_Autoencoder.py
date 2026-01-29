import torch
import json

from baselines.components.ts_processor import TsProcessor
from baselines.components.ts_detector import TsDetector
from baselines.models.detector import LSTMAutoencoder, TransformerAutoencoder


class TsAutoencoder:
    def __init__(
        self,
        app_config_path,
        checkpoint_dir,
        seq_len=512,
        batch_size=64,
        alpha="0.05",
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.alpha = alpha

        print(f"[TSMon] Using device: {self.device}")

        with open(app_config_path, "r") as f:
            self.apps = json.load(f)

        self.ts_processor = TsProcessor(seq_len)

        self.ts_model = TransformerAutoencoder().to(self.device)
        self.ts_model.load_state_dict(
            torch.load(
                f"{checkpoint_dir}/ts_transformer_lammps.pth",
                map_location=self.device
            )
        )
        self.ts_model.eval()


        self.ts_detector = TsDetector(
            model=self.ts_model,
            device=self.device,
            batch_size=batch_size,
        )

        with open(
            f"{checkpoint_dir}/ts_transformer_th_lammps.json", "r"
        ) as f:
            self.ts_threshold = json.load(f)

        print("[TSMon] Initialization complete")


    def __call__(self, df_dxt):

        chunked_sequences = self.ts_processor(df_dxt)

        for chunked_sequence in chunked_sequences:
            file_anom_score = self.ts_detector(chunked_sequence)

            if file_anom_score > self.ts_threshold["0.01"]:
                print("[TSMon][ALARM] Temporal I/O anomaly detected")
                return 1

        print("[TSMon][INFO] No temporal anomaly detected")
        return 0
