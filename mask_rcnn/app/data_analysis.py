
"""

Managing data

"""

import pandas as pd


def process_spreadsheet_data(path, type = 'fat'):

    """

    Process the extracted metrics

    :param path:
    :return:

    """

    d1 = pd.read_excel(path)
    d2 = d1[['study', 'study_id', 'liver_ct1_orig', 'liver_t2star_orig', 'liver_fat_orig', 'brunt_steatosis', 'kleiner_brunt_fibrosis', 'lobular_inflammation', 'primary_diagnosis']]

    # Select CALM cases
    d2 = d2.loc[d2['study'] == 'CALM']

    if type == 'fat':

        # Parse fat cases of interest
        d2 = d2.loc[d2['brunt_steatosis'] != 'na']
        d2 = d2.loc[d2['liver_fat_orig'] != 'na']
        d2 = d2.loc[d2['liver_fat_orig'].isnull() == False]
        d2 = d2.loc[d2['liver_fat_orig'] > 0]
        d2['liver_fat_orig'] = d2['liver_fat_orig'].astype('float')
        d2['brunt_steatosis'] = d2['brunt_steatosis'].astype('float')
        d2 = d2.loc[d2['primary_diagnosis'] == 'NAFLD']

        d2 = d2[['study', 'study_id', 'liver_fat_orig', 'brunt_steatosis', 'primary_diagnosis']]

    return d2


def process_file_data(files):

    """

    :param files:
    :return:

    """

    d2 = pd.DataFrame(columns=['study_id', 'file_name'])

    for cc, f1 in enumerate(files):

        if f1.startswith('HandE'):

            d2.loc[cc] = [f1[6:16], f1]

        else:
            d2.loc[cc] = [f1[:10], f1]

    return d2
