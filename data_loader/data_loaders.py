import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Dataset Loader 
class TimeseriesDatasetLoader(Dataset):
    def __init__(self, series, window_size, shuffle):
        self.series = torch.tensor(series.values).float()
        self.window_size = window_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.series) - self.window_size - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.shuffle:
            idx = torch.randint(len(self), size=(1,)).item()

        return (self.series[idx:idx+self.window_size], self.series[idx+self.window_size+1])


def get_loader(series, window_size, batch_size, shuffle=True):
    dataset = TimeseriesDatasetLoader(series, window_size, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# data split func
def split_data(df, split_ratio):
    _, _, y_train, y_test = train_test_split(
    df.drop('traffic', axis=1), 
    df['traffic'], 
    test_size=split_ratio, 
    shuffle=False)
    return y_train, y_test

def apply_scaler(df):
    scaler = MinMaxScaler()
    scale_cols = ['traffic']
    df['traffic'] = scaler.fit_transform(df[scale_cols])
    return df