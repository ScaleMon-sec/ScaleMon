"""
Core orchestration module of ScaleMon.
Coordinates log parsing, feature generation, inter-file and intra-file monitoring,
and forensic report generation.
"""


import torch
import json
import joblib
from scalemon.models.embedder import ResNet18FeatureExtractor
from scalemon.models.detector import DeepSVDD3, DeepSVDD4
from scalemon.models.classifier import ResNet18Classifier

from scalemon.components.log_parser import LogParser
from scalemon.components.intermon import InterMon
from scalemon.components.intramon import IntraMon
from scalemon.components.forensic_reporter import ForensicReporter

from concurrent.futures import ThreadPoolExecutor



class ScaleMon:
    def __init__(
        self,
        app_config_path,
        file_type_keywords_path,
        txt_base_dir,
        checkpoint_dir,
        num_app=3,
        alpha="0.05",
        conf_threshold=0.6,
        intra_dim=128,
        inter_detector_name = "DeepSVDD4",
        intra_detector_name = "DeepSVDD3",    
        device=None,
    ):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.alpha = alpha

        print(f"[ScaleMon] Using device: {self.device}")

        with open(app_config_path, "r") as f:
            self.apps = json.load(f)

        with open(file_type_keywords_path, "r") as f:
            self.file_type_keywords = json.load(f)

        self.preprocessor_j = {}
        self.encoder_j = {}
        self.preprocessor_f = {}

        self.inter_mon_j_model = {}
        self.inter_mon_f_model = {}
        self.intra_mon_model = {}

        self.inter_mon_j_threshold = {}
        self.inter_mon_f_threshold = {}
        self.intra_mon_threshold = {}

        self.intra_mon_embedder = ResNet18FeatureExtractor(
            output_dim=intra_dim,
            use_pretrained=True
        ).to(self.device)

        for app_id, app_dict in self.apps.items():
            app_name = app_dict["app_name"]

            self.preprocessor_j[app_id] = joblib.load(
                f"{checkpoint_dir}/inter_j_preprocessor_{app_name}.joblib"
            )
            self.encoder_j[app_id] = joblib.load(
                f"{checkpoint_dir}/inter_j_encoder_{app_name}.joblib"
            )
            self.preprocessor_f[app_id] = joblib.load(
                f"{checkpoint_dir}/inter_f_preprocessor_{app_name}.joblib"
            )

    
            assert inter_detector_name in ["DeepSVDD3", "DeepSVDD4"], \
                f"Unsupported detector: {inter_detector_name}"

            if inter_detector_name == 'DeepSVDD4':
                self.inter_mon_j_model[app_id] = DeepSVDD4(
                    app_dict["indim"]["inter_mon_j"]
                ).to(self.device)
                
                self.inter_mon_f_model[app_id] = DeepSVDD4(
                    app_dict["indim"]["inter_mon_f"]
                ).to(self.device)
                                
            else:
                self.inter_mon_j_model[app_id] = DeepSVDD3(
                    app_dict["indim"]["inter_mon_j"]
                ).to(self.device)
                
                self.inter_mon_f_model[app_id] = DeepSVDD3(
                    app_dict["indim"]["inter_mon_f"]
                ).to(self.device)
            
            
            self.inter_mon_j_model[app_id].load_state_dict(
                torch.load(
                    f"{checkpoint_dir}/inter_j_{inter_detector_name}_{app_name}.pth",
                    map_location=self.device
                )
            )
            self.inter_mon_j_model[app_id].eval()

            with open(
                f"{checkpoint_dir}/inter_j_{inter_detector_name}_th_{app_name}.json", "r"
            ) as f:
                self.inter_mon_j_threshold[app_id] = json.load(f)

            self.inter_mon_f_model[app_id].load_state_dict(
                torch.load(
                    f"{checkpoint_dir}/inter_f_{inter_detector_name}_{app_name}.pth",
                    map_location=self.device
                )
            )
            self.inter_mon_f_model[app_id].eval()

            with open(
                f"{checkpoint_dir}/inter_f_{inter_detector_name}_th_{app_name}.json", "r"
            ) as f:
                self.inter_mon_f_threshold[app_id] = json.load(f)

            assert intra_detector_name in ["DeepSVDD3", "DeepSVDD4"], \
                f"Unsupported detector: {intra_detector_name}"

            if intra_detector_name == 'DeepSVDD4':
                self.intra_mon_model[app_id] = DeepSVDD4(intra_dim).to(self.device)                  
            else:
                self.intra_mon_model[app_id] = DeepSVDD3(intra_dim).to(self.device)

            self.intra_mon_model[app_id].load_state_dict(
                torch.load(
                    f"{checkpoint_dir}/intra_{intra_detector_name}_{app_name}.pth",
                    map_location=self.device
                )
            )
            self.intra_mon_model[app_id].eval()

            with open(
                f"{checkpoint_dir}/intra_{intra_detector_name}_th_{app_name}.json", "r"
            ) as f:
                self.intra_mon_threshold[app_id] = json.load(f)


        intra_classifier = ResNet18Classifier(
            num_classes=num_app,
            pth_path=f"{checkpoint_dir}/identity_verifier.pth",
        )

 
        self.log_parser = LogParser(
            txt_base_dir,
            self.file_type_keywords,
            self.apps,
        )
        
        self.inter_mon = InterMon(
            self.preprocessor_j,
            self.encoder_j,
            self.preprocessor_f,
            self.inter_mon_j_model,
            self.inter_mon_f_model,
            self.inter_mon_j_threshold,
            self.inter_mon_f_threshold,
            self.device,
            self.alpha,
        )

        self.intra_mon = IntraMon(
            classifier=intra_classifier,
            conf_threshold=conf_threshold,
            embedder=self.intra_mon_embedder,
            detector=self.intra_mon_model,
            threshold=self.intra_mon_threshold,
            device=self.device,
            alpha=self.alpha,
        )

        self.forensic_reporter  = ForensicReporter()
        print("[ScaleMon] Initialization complete")


    def __call__(self, darshan_path):

        job_id, app_id, df_inter, df_intra = self.log_parser(darshan_path)

        if not app_id:
            print(
                "[ScaleMon][WARN] Unknown application: "
                "failed to match Darshan log to a registered app"
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_inter = executor.submit(self.inter_mon, app_id, df_inter)
            fut_intra = executor.submit(self.intra_mon, app_id, df_intra)

            job_flag, inter_anomaly_files = fut_inter.result()
            intra_anomaly_files = fut_intra.result()

        self.forensic_reporter(job_flag, inter_anomaly_files, intra_anomaly_files, job_id)