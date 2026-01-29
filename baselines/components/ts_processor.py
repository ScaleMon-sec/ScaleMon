import pandas as pd
import numpy as np
import torch

class TsProcessor():
    
    def __init__(self, chunk_len=512):
        self.chunk_len = chunk_len

    
    def make_sequence(self, df):
        files = [f for f in df['File_name'].dropna().unique()]
        sequences = []

        for file in files:
            df_file = df[(df['File_name'] == file) & (df['Module'] == 'X_POSIX')]

            if len(df_file) == 0:
                continue
            df_file = df_file.sort_values(by=['Start', 'Segment']).reset_index(drop=True)

            df_file['delta_prev'] = df_file['Start'].diff().fillna(0)
            df_file['delta_next'] = df_file['Start'].shift(-1) - df_file['Start']
            df_file['delta_next'] = df_file['delta_next'].fillna(0)
            df_file['rw_id'] = (df_file['Op'] == 'write').astype(int)

            num_cols = ['Offset', 'Length', 'delta_prev', 'delta_next']
            for c in num_cols:
                col_min = df_file[c].min()
                col_max = df_file[c].max()
                if col_max - col_min > 0:
                    df_file[c] = (df_file[c] - col_min) / (col_max - col_min)
                else:
                    df_file[c] = 0.0

            arr = df_file[['rw_id', 'Offset', 'Length', 'delta_prev', 'delta_next']].to_numpy()
            sequences.append(arr.astype(np.float32))

        return sequences
    

    def chunk_sequence(self, arr):
        
        N, F = arr.shape
        assert F == 5

        num_chunks = N // self.chunk_len
        if num_chunks == 0:
            return None

        trimmed = arr[:num_chunks * self.chunk_len]
        chunks = trimmed.reshape(num_chunks, self.chunk_len, F)
        return chunks
    
    def __call__(self, df):
        sequences = self.make_sequence(df)
        chunked_sequences = []
        for sequence in sequences:
            chunked_sequences.append(self.chunk_sequence(sequence))
        return chunked_sequences
