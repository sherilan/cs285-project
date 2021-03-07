

import argparse
import os

import tensorflow as tf
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
args = parser.parse_args()


for file in args.files:
    print(f'Processing: {file}')

    try:
        logs = []
        for e in tf.compat.v1.train.summary_iterator(file):
            for v in e.summary.value:
                logs.append(
                    dict(
                        tag=v.tag,
                        value=v.simple_value,
                        step=e.step,
                    )
                )

        df = pd.DataFrame(logs).pivot(index='step', columns='tag').sort_index()['value']
        df.columns.name = None
        df.index.name = None
        savepath = os.path.join(os.path.dirname(file), 'tb_events.h5')
        df.to_hdf(savepath, key='data')

    except Exception as e:
        print(f'Could not process: {file}')
        print(e)

print('Done')
