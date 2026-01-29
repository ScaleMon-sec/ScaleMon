"""
Generates intra-file feature representations capturing fine-grained access behaviors
within individual files.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

class Intra_FG:
    def __init__(self):
        pass
        
    def __call__(self, df_intra):
        io_images = []
       
        intra_files = []
        for fid, file in enumerate(df_intra['File_name'].dropna().unique().tolist()):
            df_file = df_intra[df_intra['File_name'] == file]
            df_file = df_file[df_file['Module'] == 'X_POSIX']
            if len(df_file) != 0:
                intra_files.append(file)
                io_images.append(self.to_npy(
                    df_file,
                    y_var="Offset",
                    value="Length",
                    size=64,
                    max_value=1.0,
                ))

        return io_images, intra_files

    def to_npy(self, df, y_var, value=None, size=256, max_value=1.0):
        x = df['Start'].to_numpy()
        y = df[y_var].to_numpy()
        op = df['Op'].str.lower().to_numpy()

        if value is not None:
            w = df[value].to_numpy()
        else:
            w = np.ones_like(x)
        x_range = (x.min(), x.max())
        y_range = (y.min(), y.max())

        img = np.zeros((2, size, size), dtype=np.float32)

        mask = (op == 'read')
        if mask.any():
            H, _, _ = np.histogram2d(
                x[mask], y[mask],
                bins=size,
                range=[x_range, y_range],
                weights=w[mask]
            )
            img[0] = H.T
            m = img[0].max()
            if m > 0:
                img[0] *= (max_value / m)

        mask = (op == 'write')
        if mask.any():
            H, _, _ = np.histogram2d(
                x[mask], y[mask],
                bins=size,
                range=[x_range, y_range],
                weights=w[mask]
            )
            img[1] = H.T
            m = img[1].max()
            if m > 0:
                img[1] *= (max_value / m)
        return img