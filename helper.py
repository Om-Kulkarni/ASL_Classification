#!/usr/bin/env python

import numpy
import random
import pandas as pd
from pathlib import Path

def proj_paths(filepaths):
    
    labels = [str(filepaths[i]).split("/")[-2] for i in range(len(filepaths))]
    
    filepaths = pd.Series(filepaths, name='Filepaths').astype(str)
    labels = pd.Series(labels, name='Label')
    
    # Concatenate filepaths and labels
    df = pd.concat([filepaths, labels], axis=1)
    
    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1).reset_index(drop = True)
    
    return df


def load_data():
    train_image_dir = Path('asl_alphabet_train')
    train_filepaths = list(train_image_dir.glob(r'**/*.jpg'))

    test_image_dir = Path('asl_alphabet_test')
    test_filepaths = list(train_image_dir.glob(r'**/*.jpg'))
    
    
    # Create df
    train_df = proj_paths(train_filepaths)
    test_df = proj_paths(test_filepaths)
    
    return (train_df['Filepaths'], train_df['Label']), (test_df['Filepaths'], test_df['Label'])


