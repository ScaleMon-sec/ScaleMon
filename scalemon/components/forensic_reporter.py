"""
Aggregates detection results and produces a structured forensic report
for post-execution analysis.
"""

import subprocess
import pandas as pd
from typing import List, Optional

class ForensicReporter:
    def __init__(self, extended_fields: Optional[List[str]] = None):
        DEFAULT_SLURM_FIELDS = [
            "JobID",
            "User",
            "UID",
            "JobName",
            "Partition",
            "NodeList",
            "State",
            "ExitCode",
            "Submit",
            "Start",
            "End",
            "SubmitLine",
            "WorkDir",
        ]

        self.slurm_fields = DEFAULT_SLURM_FIELDS
        if extended_fields:
            self.slurm_fields.extend(extended_fields)


    def _query_slurm(self, job_id: int) -> dict:
        fmt = ",".join(self.slurm_fields)
        cmd = ["sacct", "-j", str(job_id), "--format", fmt, "--parsable2"]

        try:
            result = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True
            )
        except Exception:
            return {}

        lines = result.strip().split("\n")
        if len(lines) < 2:
            return {}

        keys = lines[0].split("|")
        values = lines[1].split("|")

        return dict(zip(keys, values))


    def __call__(
        self,
        job_flag: bool,
        inter_anomaly_files: List[str],
        intra_anomaly_files: List[str],
        job_id: int
    ):
        slurm_meta = self._query_slurm(job_id)

        rows = []

        for files in inter_anomaly_files:
            rows.append({
                "Job_ID": job_id,
                "User": slurm_meta.get("User", "unknown"),
                "Detection Level": "Inter-file",
                "File_Path": files,
                "Job_Anomaly": job_flag
            })

        for files in intra_anomaly_files:
            rows.append({
                "Job_ID": job_id,
                "User": slurm_meta.get("User", "unknown"),
                "Detection Level": "Intra-file",
                "File_Path": files,
                "Job_Anomaly": job_flag
            })

        df = pd.DataFrame(rows)

        self._print_report(df, slurm_meta)

        return df

    def _print_report(self, df: pd.DataFrame, slurm_meta: dict):
        print("\n" + "=" * 80)
        print(" ScaleMon Forensic Report")
        print("=" * 80)

        if slurm_meta:
            print("[ Job Context ]")
            for k, v in slurm_meta.items():
                print(f"  {k:12s}: {v}")
        else:
            print("[ Job Context ] unavailable")

        print("\n[ Detected Anomalies ]")
        if df.empty:
            print("  No anomalies detected.")
        else:
            print(df.to_string(index=False))

        print("=" * 80 + "\n")
