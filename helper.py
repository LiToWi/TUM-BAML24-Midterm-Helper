import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Helper:
    def __init__(self, dataset_src, delimiter=None):
        self.dataset_src = dataset_src
        self.dataset = pd.read_csv(dataset_src, delimiter=delimiter)

    ### Manage the dataset ###
    def get_dataset(self) -> pd.DataFrame:
        return self.dataset
    def set_dataset(self, dataset):
        self.dataset = dataset
    def reset_dataset(self) -> pd.DataFrame:
        self.dataset = self.OG_DATASET
        return self.dataset
    
    ### Visualize the dataset ###
    def print_head(self, *args, **kwargs):
        print(self.dataset.head(*args, **kwargs))

    def pretty_plot(self, *args, **kwargs):
        self.dataset.plot(*args, **kwargs)
        plt.show()


if __name__ == '__main__':
    users_helper = Helper('users.csv', delimiter=';')
    users_helper.pretty_plot(x='uid', y='p', kind='scatter')