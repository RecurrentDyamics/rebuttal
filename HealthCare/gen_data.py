import argparse
import numpy as np
import pandas as pd
import ujson as json

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, default = 'train')
args = parser.parse_args()

df = pd.read_csv('./raw.csv')
df = df.drop(df.columns[0], axis = 1)

desc_cols = ['MechVent', 'DiasABP', 'HR', 'Na', 'Cholesterol', 'FiO2', 'PaO2', 'WBC', 'pH', 'Albumin', 'Glucose', 'SaO2', 'Temp', 'AST', 'HCO3', 'BUN', 'Bilirubin', 'RespRate', 'Mg', 'HCT', 'SysABP', 'NIDiasABP', 'K', 'TroponinT', 'GCS', 'Lactate', 'NISysABP', 'Creatinine', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP', 'ALT', 'ALP']

mean = df[desc_cols].mean()
std = df[desc_cols].std()

df[desc_cols] = (df[desc_cols] - mean) / std

df = df.groupby('id')
ids = df.groups.keys()

train_ids = ids[:3200]
test_ids = ids[-800:]

def gen_sequence(values, masks, dir_):
    assert(dir_ in ['forward', 'backward'])
    if dir_ == 'backward':
        values, masks = values[::-1], masks[::-1]

    ret = {'values': values, 'masks': masks}

    deltas, lasts = [], []

    n, d = len(values), len(values[0])

    for i in range(len(values)):
        if i == 0:
            delta = np.zeros(d)
            last = np.zeros(d)
        else:
            delta = np.ones(d) * np.asarray(masks[i - 1]) + np.asarray(deltas[i - 1]) * (1 - np.asarray(masks[i - 1]))
            last = np.asarray(values[i - 1]) * np.asarray(masks[i - 1]) + np.asarray(lasts[i - 1]) * (1 - np.asarray(masks[i - 1]))

        deltas.append(delta.tolist())
        lasts.append(last.tolist())

    ret.update({'deltas': deltas, 'lasts': lasts})
    return ret

def run(ids, output_file):
    fs = open(output_file, 'w')

    for id_ in ids:
        print 'Processing ID {}'.format(id_)
        patient_df = df.get_group(id_)

        values, masks = [], []
        label = patient_df['label'].iloc[0]

        patient_df = patient_df.groupby('hour')
        for hour in range(48):
            value = patient_df.get_group(hour)[desc_cols]
            mask = ~pd.isnull(value)

            value = value.fillna(0.0)

            value = value.as_matrix()[0]
            mask = mask.as_matrix()[0]

            values.append(value.tolist())
            masks.append(map(lambda x: 1 if x else 0, mask.tolist()))

        forward = gen_sequence(values, masks, 'forward')
        backward = gen_sequence(values, masks, 'backward')

        ret_d = {'forward': forward, 'backward': backward, 'label': label}
        ret_d.update({'is_train': args.task == 'train'})

        ret_s = json.dumps(ret_d)

        fs.write(ret_s + '\n')

    fs.close()

if __name__ == '__main__':
    if args.task == 'train':
        run(train_ids, output_file = './data/train')
    elif args.task == 'test':
        run(test_ids, output_file = './data/test')


