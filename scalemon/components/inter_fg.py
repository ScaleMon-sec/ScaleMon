"""
Generates inter-file feature representations from parsed I/O traces
for anomaly detection models.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Inter_FG:
    def __init__(self, preprocessor_j, encoder_j, preprocessor_f):
        self.preprocessor_j = preprocessor_j
        self.encoder_j = encoder_j
        self.preprocessor_f = preprocessor_f
        
    def __call__(self, app_id, df_inter):
        EPV_dict = {
            "job": None,
            "file": None
        }
        EPV_dict['job'] = self.preprocessor_j[app_id].transform(self.one_hot_sum_file_types(self.aggregate_df(df_inter), encoder=self.encoder_j[app_id], fit=False)[0])
        EPV_dict['file'] = self.preprocessor_f[app_id].transform(df_inter)
        inter_files = df_inter['file_path'].tolist()

        return EPV_dict, inter_files
    
    def aggregate_df(self, df, group_key="job_id"):

        agg_df = df.groupby(group_key).agg({
            "file_type": lambda x: list(x), 
            "min_dist": "min",
            "mean_dist": "mean",
            "max_dist": "max",
            "runtime": "first",
        }).reset_index()

        return agg_df

    def one_hot_sum_file_types(self, df, encoder=None, fit=False):
        exploded = df.explode("file_type")[["job_id", "file_type"]]

        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        if fit:
            onehots = encoder.fit_transform(exploded[["file_type"]])
        else:
            onehots = encoder.transform(exploded[["file_type"]])

        onehot_df = pd.DataFrame(onehots, columns=encoder.get_feature_names_out(["file_type"]))

        exploded = pd.concat([exploded.reset_index(drop=True), onehot_df], axis=1)
        summed = exploded.groupby("job_id").sum().reset_index()

        df_out = df.drop(columns=["file_type"]).merge(summed, on="job_id", how="left")

        return df_out, encoder



