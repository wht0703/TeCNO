import numpy as np
import pandas as pd


def median_frequency_weights(dataframes, class_num):
    df = pd.read_csv(dataframes)
    frequency = [0] * class_num
    for i in range(0, len(frequency)):
        frequency[i] = len(df[df["class"] == i])
    median = np.median(frequency)
    print(np.sum(frequency))
    weights = [float(median / j) for j in frequency]
    return weights

if __name__ == '__main__':
    print(median_frequency_weights('../../dataframes/cataract_split_250px_5fps.csv', 13))
