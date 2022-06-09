from tqdm import trange

import numpy as np
import os
import pandas as pd


class Preprocess:

    def __init__(self, raw_metadata_path, preprocessed_metadata_path=None):

        print('Start preprocessing: {}.'.format(raw_metadata_path))

        self.raw_metadata_path = raw_metadata_path

        raw_p = os.path.normpath(self.raw_metadata_path).split(os.sep)
        self.dir_path = os.path.join(*raw_p[:-1])

        if preprocessed_metadata_path is None:
            file_split = "_".join(raw_p[-1].split("_")[-2:])
            file_split = ".".join(file_split.split(".")[:-1])
            self.preprocessed_metadata_path = os.path.join(self.dir_path, "preprocessed_{}.csv".format(file_split))
        else:
            self.preprocessed_metadata_path = preprocessed_metadata_path

        if not os.path.isfile(self.preprocessed_metadata_path):
            self.reformat_csv()

        print('Finished preprocessing: {}.'.format(self.preprocessed_metadata_path))

    def reformat_csv(self):
        raw_df = pd.read_csv(self.raw_metadata_path, sep='\t')
        df = raw_df.copy()

        for i in trange(len(df)):
            df.loc[i, ["path"]] = os.path.join("./clips", df["path"][i])
            
        df.to_csv(self.preprocessed_metadata_path)


if __name__ == '__main__':
    test_preprocess = Preprocess("./data/common_voice_yue/test.tsv")
    train_preprocess = Preprocess("./data/common_voice_yue/train.tsv")
    valid_preprocess = Preprocess("./data/common_voice_yue/dev.tsv")