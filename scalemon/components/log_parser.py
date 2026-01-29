"""
Parses persisted I/O traces and extracts structured job- and file-level metadata
used by downstream analysis components.
"""

import numpy as np
import pandas as pd
import os
import subprocess
from itertools import combinations
import re
from concurrent.futures import ThreadPoolExecutor
    
class LogParser:
    def __init__(self, txt_base_dir, file_type_keywords_dict, app_config_dict):
        self.txt_base_dir = txt_base_dir

        self.file_type_keywords = {
            k: set(v) for k, v in file_type_keywords_dict.items()
        }

        self.app_config = app_config_dict

    def run_inter(self, darshan_path, base_name):
        darshan_out = os.path.join(self.txt_base_dir, f"{base_name}.inter.txt")
        subprocess.run(
            f"darshan-parser --show-incomplete {darshan_path} > {darshan_out}",
            shell=True,
            check=True
        )
        job_id, app_id, df_inter = self.process_inter_txt(darshan_out)
        return job_id, app_id, df_inter


    def run_intra(self, darshan_path, base_name):
        dxt_out = os.path.join(self.txt_base_dir, f"{base_name}.intra.txt")
        subprocess.run(
            f"darshan-dxt-parser --show-incomplete {darshan_path} > {dxt_out}",
            shell=True,
            check=True
        )
        df_intra = self.process_intra_txt(dxt_out)
        return df_intra


    def __call__(self, darshan_path):
        base_name = os.path.splitext(os.path.basename(darshan_path))[0]

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_inter = executor.submit(self.run_inter, darshan_path, base_name)
            f_intra = executor.submit(self.run_intra, darshan_path, base_name)

            job_id, app_id, df_inter = f_inter.result()
            df_intra = f_intra.result()

        return job_id, app_id, df_inter, df_intra

    def get_app_id(self, exe_name):
        app_id = None
        for id, app_dict in self.app_config.items():
            for v in app_dict['keywords']:
                if v.lower() in exe_name.lower():
                    app_id = id
        return app_id
    
    def lca_distance(self, file1, file2):
        parts1 = [p for p in file1.split('/') if p]
        parts2 = [p for p in file2.split('/') if p]
        common = 0
        for a, b in zip(parts1, parts2):
            if a == b:
                common += 1
            else:
                break
        return (len(parts1) - common) + (len(parts2) - common)

    def compute_distances(self, file_list):
        if len(file_list) == 1:
            f = file_list[0]
            return {f: (0,0,0)}
        distances = {f: [] for f in file_list}
        for f1, f2 in combinations(file_list, 2):
            d = self.lca_distance(f1, f2)
            distances[f1].append(d)
            distances[f2].append(d)
        stats = {}
        for f, dlist in distances.items():
            if not dlist:
                stats[f] = (0,0,0)
            else:
                stats[f] = (np.min(dlist), np.mean(dlist), np.max(dlist))
        return stats

    def parse_darshan(self, file_path):
        import os
        import re
        import pandas as pd

        exe_str = None
        start_time = None
        end_time = None
        job_id = None

        tables = []
        current_header = None
        current_rows = []

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('# start_time:'):
                    try:
                        start_time = int(line.split(':', 1)[1].strip())
                    except:
                        start_time = None

                elif line.startswith('# end_time:'):
                    try:
                        end_time = int(line.split(':', 1)[1].strip())
                    except:
                        end_time = None

                elif line.startswith('# jobid:'):
                    try:
                        job_id = int(line.split(':', 1)[1].strip())
                    except:
                        job_id = None

                elif line.startswith('# exe:'):
                    exe_str = line.split('# exe:', 1)[1].strip()

                elif line.startswith('#<module>'):
                    if current_header and current_rows:
                        df_block = pd.DataFrame(current_rows, columns=current_header)
                        tables.append(df_block)

                    current_header = re.split(r'\t+', line.lstrip('#'))
                    current_rows = []

                elif current_header and line and not line.startswith('#'):
                    parts = re.split(r'\t+', line)
                    if len(parts) == len(current_header):
                        current_rows.append(parts)

            if current_header and current_rows:
                df_block = pd.DataFrame(current_rows, columns=current_header)
                tables.append(df_block)

        if tables:
            df = pd.concat(tables, ignore_index=True)
        else:
            df = pd.DataFrame()

        if exe_str:
            exe_name = os.path.basename(exe_str.split()[0])
        else:
            exe_name = None

        return df, job_id, exe_name, start_time, end_time

    
    def extract_read_write_files(self, df):
        reads_pattern = re.compile(r"_READS$")
        writes_pattern = re.compile(r"_WRITES$")
        reads = set()
        writes = set()
        if df.empty or not {'<counter>','<value>','<file name>'}.issubset(df.columns):
            return reads, writes

        for _, row in df.iterrows():
            counter = str(row['<counter>']).strip()
            value = row['<value>']
            filename = str(row['<file name>']).strip()

            if reads_pattern.search(counter) and value != "0":
                reads.add(filename)
            elif writes_pattern.search(counter) and value != "0":
                writes.add(filename)
        return reads, writes

    def file_type_refiner(self, s: str) -> str:
        if not s or not isinstance(s, str):
            return "unknown"
    
        tokens = re.split(r'[._]', s)

        token_set = set(tokens) 

        for key, value in self.file_type_keywords.items():
            if token_set & value:
                return key

        tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
        
        if tokens:
            return tokens[-1].strip()
        else:
            return "unknown"


    def process_inter_txt(self, file_path):
        df_raw, job_id, exe_name, start_time, end_time = self.parse_darshan(file_path)
        app_id = self.get_app_id(exe_name)

        if start_time is not None and end_time is not None:
            runtime = end_time - start_time
        else:
            runtime = None

        reads, writes = self.extract_read_write_files(df_raw)

        all_files = list(reads.union(writes))
        all_files = [f for f in all_files if not (f.startswith("<") and f.endswith(">"))]

        if not all_files:
            return job_id, app_id, pd.DataFrame(columns=[
                "job_id","file_id","file_path","file_type",
                "read","write","min_dist","mean_dist","max_dist","runtime"
            ])

        stats = self.compute_distances(all_files)
        records = []
            
        for idx, f in enumerate(all_files): 
            is_read = f in reads
            is_write = f in writes
            file_path_str = f
            file_type = self.file_type_refiner(os.path.basename(f))
            min_d, med_d, max_d = stats[f]
            file_id = f"{job_id}_{idx}"
            
            records.append([
                job_id,
                file_id,
                file_path_str,
                file_type,
                is_read,
                is_write,
                min_d,
                med_d,
                max_d,
                runtime
            ])

        return job_id, app_id, pd.DataFrame(
            records,
            columns=[
                "job_id","file_id","file_path","file_type",
                "read","write","min_dist","mean_dist","max_dist","runtime"
            ]
        )

    def process_intra_txt(self, log_file_path):
        columns = [
            'Module', 'Rank', 'Wt/Rd', 'Segment',
            'Offset', 'Length', 'Start(s)', 'End(s)',
            'File_id', 'File_name'
        ]
        rows = []

        current_file_id = None
        current_file_name = None

        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()

                    if stripped_line.startswith('# DXT, file_id:'):
                        try:
                            parts = stripped_line.split(',')
                            current_file_id = parts[1].split(':')[1].strip()
                            current_file_name = parts[2].split(':')[1].strip()
                        except IndexError:
                            continue

                    elif stripped_line.startswith('X_POSIX') or stripped_line.startswith('X_MPIIO'):
                        parts = stripped_line.split()
                        if len(parts) >= 8:
                            entry = parts[:8] + [current_file_id, current_file_name]
                            rows.append(entry)

        except FileNotFoundError:
            print(f"Error: The file '{log_file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred while reading the log file: {e}")
            return None

        if not rows:
            print("No 'X_POSIX' or 'X_MPIIO' data rows found in the log file.")
            return None

        df = pd.DataFrame(rows, columns=columns)

        df['Rank'] = df['Rank'].astype(int)
        df['Segment'] = df['Segment'].astype('uint64')
        df['Offset'] = df['Offset'].astype('uint64')
        df['Length'] = df['Length'].astype('uint64')
        df['Start(s)'] = df['Start(s)'].astype(float)
        df['End(s)'] = df['End(s)'].astype(float)

        return df