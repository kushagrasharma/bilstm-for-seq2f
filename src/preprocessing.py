import pandas as pd
import numpy as np

function2idx = {"negative": 0, "ferritin": 1, "gpcr": 2, "p450": 3, "protease": 4}

input_dir = '../data/raw/'
data_dir = '../data/processed/'
max_seq_len = 800


def read_and_concat_data():
    df_cysteine = pd.read_csv(input_dir + 'uniprot-cysteine+protease+AND+reviewed_yes.tab', sep='\t', skiprows=(0),
                              header=(0))
    df_cysteine.drop(['Entry name', "Status"], axis=1, inplace=True)
    df_cysteine.columns = ['id', 'sequence']
    df_cysteine['function'] = function2idx['protease']

    df_serine = pd.read_csv(input_dir + 'uniprot-serine+protease+AND+reviewed_yes.tab', sep='\t', skiprows=(0),
                            header=(0))
    df_serine.drop(['Entry name', "Status"], axis=1, inplace=True)
    df_serine.columns = ['id', 'sequence']
    df_serine['function'] = function2idx['protease']

    df_gpcr = pd.read_csv(input_dir + 'uniprot-gpcr+AND+reviewed_yes.tab', sep='\t', skiprows=(0), header=(0))
    df_gpcr.drop(['Entry name', "Status"], axis=1, inplace=True)
    df_gpcr.columns = ['id', 'sequence']
    df_gpcr['function'] = function2idx['gpcr']

    df_p450 = pd.read_csv(input_dir + 'uniprot-p450+AND+reviewed_yes.tab', sep='\t', skiprows=(0), header=(0))
    df_p450.drop(['Entry name', "Status"], axis=1, inplace=True)
    df_p450.columns = ['id', 'sequence']
    df_p450['function'] = function2idx['p450']

    df_f = pd.read_csv(input_dir + 'uniprot-ferritin-filtered-reviewed_yes.tab', sep='\t', skiprows=(0), header=(0))
    df_f.drop(['Entry name', "Status"], axis=1, inplace=True)
    df_f.columns = ['id', 'sequence']
    df_f['function'] = function2idx['ferritin']

    df_positive = pd.concat([df_cysteine, df_serine, df_f, df_gpcr, df_p450], ignore_index=True)
    duplicates = list(df_positive[df_positive.duplicated('id')].id)

    df_uniprot = pd.read_csv(input_dir + 'uniprot-reviewed_yes.tab', sep='\t', skiprows=(0), header=(0))
    df_uniprot = df_uniprot.drop(["Entry name", "Status", "Gene names", "Gene ontology (molecular function)",
                                  "Gene ontology IDs", "Gene ontology (cellular component)",
                                  "Gene ontology (biological process)", "Gene ontology (GO)"], axis=1)
    df_uniprot['function'] = function2idx['negative']
    df_uniprot.columns = ['id', 'sequence', 'function']
    df_uniprot[~df_uniprot.id.isin(duplicates)]

    df_all = pd.concat([df_uniprot, df_positive], ignore_index=True)
    df_all.sort_values(by='function', inplace=True, ascending=False)
    df_all = df_all.drop_duplicates(subset='id').reset_index(drop=True)

    print("Finished reading raw data and concating")

    return df_all


def clean_sequence_length(dataframe):
    # Add 800 amino acids from C-terminus for the longest proteins
    reverse_rows = []
    for index, row in dataframe[dataframe.sequence.apply(len) > max_seq_len].iterrows():
        reverse_rows.append([row.id + '_r', row.sequence[::-1], row.function])

    reverse_rows = pd.DataFrame(reverse_rows, columns=['id', 'sequence', 'function'])
    dataframe = pd.concat([dataframe, reverse_rows], ignore_index=True)

    # Cut all sequences to 800 char
    dataframe['sequence'] = dataframe.sequence.apply(lambda x: x[:max_seq_len])

    dataframe['length'] = dataframe.sequence.apply(len)
    dataframe = dataframe.sort_values(by='length').reset_index(drop=True)

    print("Finished cleaning sequences by length")

    return dataframe


df = read_and_concat_data()

df = clean_sequence_length(df)

np.savetxt(data_dir + 'sequence.txt', df.sequence.values, fmt='%s')
np.savetxt(data_dir + 'function.txt', df.function.values, fmt='%s')

print("Saved sequence and function to txt")