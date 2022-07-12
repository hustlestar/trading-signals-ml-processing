import pandas as pd


def min_max_scaler(f):
    # normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(f)
    return pd.DataFrame(scaled_values, columns=f.columns)


def standard_scaler(f):
    # standardization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(f)
    return pd.DataFrame(scaled_values, columns=f.columns)