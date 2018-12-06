import pandas as pd
import numpy as np


def process_output_csv(pred, losses, decoder_input_columns, clothe, clothe_type_count, sort=True):
    con_dim = len(decoder_input_columns)
    output_columns = []
    for i in range(1, 5):
        output_columns += [f'染料_{i}', f'濃度_{i}']
    output_columns += ['布號', 'delta_lab']
    result = pd.DataFrame(columns=output_columns)

    for i, con in enumerate(pred[:, :con_dim - clothe_type_count]):
        dye_idx = np.where(con > 0)
        dyes = decoder_input_columns[dye_idx]
        dye_cons = con[dye_idx]
        for j in range(len(dye_idx[0])):
            result.loc[i, [f'染料_{j+1}', f'濃度_{j+1}']] = [dyes[j].split('_')[1], f'{dye_cons[j]:.4f}']

    result['布號'] = clothe
    result['delta_lab'] = [f"{x:.4f}" for x in losses]
    result.delta_lab = result.delta_lab.astype(float).fillna(0.0)
    if sort:
        result = result.sort_values(by=['delta_lab'])
    return result
