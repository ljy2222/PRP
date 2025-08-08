import numpy as np
import pandas as pd


def compute_long_view(row):
    if row['duration_ms'] <= 18000:
        return 1 if row['play_time_ms'] >= row['duration_ms'] else 0
    else:
        return 1 if row['play_time_ms'] >= 18000 else 0


if __name__ == '__main__':
    data_path = '/root/Test/KDD/Scenario-Wise-Rec_1/scripts/data/kuairand'
    data = pd.read_csv(data_path + '/kuairand_sample.csv')

    data['long_view'] = np.where(
        data['duration_ms'] <= 18000,
        np.where(data['play_time_ms'] >= data['duration_ms'], 1, 0),
        np.where(data['play_time_ms'] >= 18000, 1, 0)
    )
    print(data['is_click'].value_counts())
    print(data['long_view'].value_counts())
    print(data['tab'].value_counts())

    new_data_path = './data/kuairand/kuairand_multi_task_sample.csv'
    data.to_csv(new_data_path, index=False)
