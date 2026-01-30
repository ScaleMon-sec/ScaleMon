# ScaleMon

ScaleMon is a scalable attack detection framework for High-Performance Computing (HPC) environments.
It analyzes hierarchical I/O behaviors of HPC jobs to identify abnormal execution patterns using a combination of
rule-based checks and machine learning based anomaly detection.

This repository provides an offline analysis pipeline that operates on persisted I/O traces
and produces structured forensic reports for post-execution investigation.

---

## Repository Structure

'''text
.
├── baselines/          # Time-series based autoencoder baselines used for comparison
│   ├── components/
│   └── models/
├── configs/            # Application and file-type configuration files
├── engine/             # Entry points for single and parallel analysis
├── scalemon/           # Core ScaleMon implementation
│   ├── components/     # Parsing, feature generation, and monitoring logic
│   ├── models/         # Classifiers, embedders, and anomaly detectors
│   └── utils/          # Visualization and helper utilities
└── README.md
'''
---

## Usage

ScaleMon supports both single-trace analysis and scalable multi-trace analysis.

### Analyze a Single Trace

    python3 -m engine.scalemon --darshan_pth "<darshan_trace_path>"

This command analyzes a single I/O trace and outputs a forensic report to standard output.

---

### Analyze Multiple Traces in Parallel

    srun -N <num_nodes> -n <num_tasks> \
      python3 -m engine.scalemon --darshan_dir "<darshan_traces_dir>" \
      >> forensic_report.txt

This mode is designed for large-scale analysis of multiple traces using distributed resources.
All generated forensic reports are appended to a single output file.

---

## Forensic Report Format

ScaleMon produces a structured forensic report for each analyzed job.

Example format:

    ================================================================================
     ScaleMon Forensic Report
    ================================================================================
    [ Job Context ]
      JobID       : <job_id>
      User        : <user>
      JobName     : <job_name>
      State       : <state>
      Start       : <start_time>
      End         : <end_time>

    [ Detected Anomalies ]
      Job_ID  User Detection Level            File_Path              Job_Anomaly
      <id>    <u>  Intra-file / Inter-file    <abstract_file_path>   <True|False>
    ================================================================================

---

## Detection Pipeline Overview

ScaleMon applies hierarchical monitoring at two levels.

### Inter-file Monitoring

Detects anomalous job-level and file-level access patterns using statistical features and
machine learning based anomaly detection.

### Intra-file Monitoring

Analyzes fine-grained access behaviors within individual files by transforming I/O patterns
into compact representations suitable for learning based detection.

A rule-based pre-filter is applied before machine learning models to capture obvious violations
and reduce false positives.

---

## Baselines

The baselines/ directory contains time-series based autoencoder models used for experimental comparison.
These models operate on sequential I/O features and are independent of the hierarchical ScaleMon pipeline.

---

## Notes

This repository focuses on offline analysis of persisted I/O traces.

System-specific deployment details and datasets are intentionally excluded.

All identifiers and paths should be treated as trace-derived metadata.

---

## Disclaimer

This code is provided for research purposes only.
Any environment-specific details have been abstracted to preserve anonymity and reproducibility.
