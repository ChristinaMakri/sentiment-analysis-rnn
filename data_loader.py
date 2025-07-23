import pandas as pd
import os

def load_sst_data(sst_dir):
    sentences_df = pd.read_csv(os.path.join(sst_dir, "datasetSentences.txt"), sep='\t')
    split_df = pd.read_csv(os.path.join(sst_dir, "datasetSplit.txt"), sep=',')
    dictionary_df = pd.read_csv(os.path.join(sst_dir, "dictionary.txt"), sep='|', header=None)
    sentiment_df = pd.read_csv(os.path.join(sst_dir, "sentiment_labels.txt"), sep='|')

    dictionary_df.columns = ['phrase', 'id']

    sentences_df['sentence_index'] = sentences_df['sentence_index'].astype(int)
    dictionary_df['id'] = dictionary_df['id'].astype(int)

    merged = sentences_df.merge(split_df, on='sentence_index')
    merged = merged.merge(dictionary_df, left_on='sentence_index', right_on='id')
    merged = merged.merge(sentiment_df, left_on='sentence_index', right_on='phrase ids')

    return merged
