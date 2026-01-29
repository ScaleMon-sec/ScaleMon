"""
Entry point for parallel multi-trace analysis.
Distributes independent I/O trace analyses across multiple processes or nodes
and aggregates forensic reports.
"""

import argparse
import os
from pathlib import Path

import torch
from mpi4py import MPI

from scalemon.components.scalemon import ScaleMon


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ScaleMon multi-job processing with MPI"
    )

    parser.add_argument(
        "--darshan_dir",
        type=Path,
        required=True,
        help="Directory containing darshan files"
    )

    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project base directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device"
    )

    parser.add_argument(
        "--app_config",
        type=Path,
        default=Path("configs/app_config.json"),
        help="Application config JSON"
    )
    parser.add_argument(
        "--file_type_keywords",
        type=Path,
        default=Path("configs/file_type_keywords.json"),
        help="File type keyword JSON"
    )
    parser.add_argument(
        "--txt_base_dir",
        type=Path,
        default=Path("data/text/multi_job"),
        help="Output text directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory"
    )

    parser.add_argument("--num_app", type=int, default=3)
    parser.add_argument("--alpha", type=str, default="0.05")
    parser.add_argument("--conf_threshold", type=float, default=0.6)
    parser.add_argument("--intra_dim", type=int, default=128)
    parser.add_argument("--inter_detector", type=str, default="DeepSVDD4")
    parser.add_argument("--intra_detector", type=str, default="DeepSVDD3")

    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_dir = args.base_dir

    app_config_path = base_dir / args.app_config
    file_type_keywords_path = base_dir / args.file_type_keywords
    darshan_dir = base_dir / args.darshan_dir
    txt_base_dir = base_dir / args.txt_base_dir
    checkpoint_dir = base_dir / args.checkpoint_dir

    if rank == 0:
        os.makedirs(txt_base_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    comm.Barrier()

    device = resolve_device(args.device)

    if rank == 0:
        print(f"[ScaleMon] Running with {size} MPI ranks on device={device}")

    scale_mon = ScaleMon(
        app_config_path=app_config_path,
        file_type_keywords_path=file_type_keywords_path,
        txt_base_dir=txt_base_dir,
        checkpoint_dir=checkpoint_dir,
        num_app=args.num_app,
        alpha=args.alpha,
        conf_threshold=args.conf_threshold,
        intra_dim=args.intra_dim,
        inter_detector_name=args.inter_detector,
        intra_detector_name=args.intra_detector,
        device=device,
    )

    darshan_files = sorted(darshan_dir.glob("*.darshan"))

    my_files = darshan_files[rank::size]

    if rank == 0:
        print(f"[ScaleMon] Total files: {len(darshan_files)}")

    for f in my_files:
        scale_mon(f)

if __name__ == "__main__":
    main()