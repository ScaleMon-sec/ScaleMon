"""
Implements inter-file monitoring.
Analyzes job-level and file-level I/O access patterns to detect anomalous behaviors.
"""

import torch
from scalemon.components.inter_fg import Inter_FG

class InterMon:
    def __init__(
        self,
        preprocessor_j,
        encoder_j,
        preprocessor_f,
        j_model,
        f_model,
        j_threshold,
        f_threshold,
        device="cuda",
        alpha=0.95,
    ):

        self.inter_fg = Inter_FG(
            preprocessor_j = preprocessor_j,
            encoder_j = encoder_j,
            preprocessor_f = preprocessor_f
        )
                
        self.j_model = j_model
        self.f_model = f_model
        self.j_threshold = j_threshold
        self.f_threshold = f_threshold
        self.device = device
        self.alpha = alpha

        for m in self.j_model.values():
            m.to(self.device).eval()
        for m in self.f_model.values():
            m.to(self.device).eval()


    def phase1(self, file_tensor):
        return (file_tensor[:, :-5] == 0).all(dim=1)


    @torch.no_grad()
    def phase2(self, app_id, job_tensor, file_tensor, pass_idx):
        B = file_tensor.size(0)
        device = file_tensor.device

        j_score = self.j_model[app_id](job_tensor)
        job_flag = (
            j_score > self.j_threshold[app_id][self.alpha]
        ).any().item()


        file_mask = torch.zeros(B, dtype=torch.bool, device=device)

        if pass_idx.numel() > 0:
            f_score = self.f_model[app_id](file_tensor[pass_idx])
            phase2_mask = (
                f_score > self.f_threshold[app_id][self.alpha]
            )
            file_mask[pass_idx] = phase2_mask

        return job_flag, file_mask


    def __call__(self, app_id, df_inter):
        
        EPV_dict, inter_files = self.inter_fg(
            app_id, df_inter
        ) 
           
        job_tensor = torch.from_numpy(
            EPV_dict["job"]
        ).float().to(self.device)

        file_tensor = torch.from_numpy(
            EPV_dict["file"]
        ).float().to(self.device)
 
        job_tensor = (
            torch.from_numpy(EPV_dict["job"])
            .float()
            .to(self.device)
        )
        file_tensor = (
            torch.from_numpy(EPV_dict["file"])
            .float()
            .to(self.device)
        )

        phase1_mask = self.phase1(file_tensor)
        pass_idx = (~phase1_mask).nonzero(as_tuple=True)[0]

        job_flag, phase2_mask = self.phase2(
            app_id,
            job_tensor,
            file_tensor,
            pass_idx,
        )
        final_file_mask = phase1_mask | phase2_mask

        anomaly_indices = final_file_mask.nonzero(as_tuple=True)[0].tolist()

        inter_anomaly_files = [inter_files[idx] for idx in anomaly_indices]
        
        return job_flag, inter_anomaly_files