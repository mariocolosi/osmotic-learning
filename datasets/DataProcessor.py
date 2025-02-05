# v0.1.0

import re
import pandas as pd
import torch
import matplotlib.pyplot as plt
import itertools


class DataProcessor:
    def __init__(self, window_size = 10):
        """
        Initialize the DataProcessor with window size and agent data.
        """
        self.data = None
        self.window_size = window_size


    def load_data(self, train_data, test_data):
        """
        Load agent data.
        """
        data = {
            "train": train_data,
            "test":  test_data
        }
        self.data = data
        return data


    def load_data_from_csv(self, train_csv, test_csv):
        """
        Load agent data from CSV files.
        """
        data = {
            "train": self.convert_csv_to_data(train_csv),
            "test":  self.convert_csv_to_data(test_csv)
        }
        self.data = data
        return data


    def convert_csv_to_data(self, csv_file):
        """
        Convert CSV file to dictionary of agent data.
        """
        df = pd.read_csv(csv_file)
        agents = {}
        pattern = r"Agent([A-Za-z0-9]+)\[(\d+)\]"
        
        for col in df.columns:
            match = re.match(pattern, col)
            if match:
                agent_id, series_index_str = match.groups()  # agent_id rimane stringa
                series_index = int(series_index_str)
                series_values = df[col].dropna().tolist()
                
                if agent_id not in agents:
                    agents[agent_id] = {}
                agents[agent_id][series_index] = series_values

        result = {}
        for agent_id, series_dict in agents.items():
            series_list = [series for _, series in sorted(series_dict.items())]
            result[agent_id] = series_list

        return result


    def save_data_to_csv(self, data_dict, file_path):
        """
        Save agent data to CSV files.
        """
        columns = {}
        for client_id, series_list in data_dict.items():
            for idx, series in enumerate(series_list):
                col_name = f"Agent{client_id}[{idx}]"
                columns[col_name] = series
        
        max_length = max((len(series) for series in columns.values()), default=0)
        
        for col_name, series in columns.items():
            if len(series) < max_length:
                columns[col_name] = series + [None] * (max_length - len(series))
        
        df = pd.DataFrame(columns)
        df.to_csv(file_path, index=False)


    def plot_time_series(self, marker_interval=10, figsize=(8, 6), fontsize=10):
        """
        Plot time series data for all agents.
        """
        plt.rcParams.update({'font.size': fontsize})
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        markers = itertools.cycle(['s', 'o', 'x', 'D', '^', 'v', '<', '>', 'p', '*', 'h'])
        
        titles = ['(a) Training data', '(b) Testing data']
        splits = ['train', 'test']
        
        handles = []
        labels = []
        
        for ax, split, title in zip(axes, splits, titles):
            for label, features in self.data[split].items():
                for i, y_values in enumerate(features):
                    x = range(len(y_values))
                    marker = next(markers)
                    line, = ax.plot(
                        x, y_values,
                        label=fr"$\text{{Agent}}_{{{label}}}[{i}]$",
                        linewidth=1,
                        marker=marker,
                        markevery=max(1, len(x) // marker_interval), 
                        markersize=16
                    )
                    handles.append(line)
                    labels.append(fr"$\text{{Agent}}_{{{label}}}[{i}]$")

            ax.set_title(title, fontsize=fontsize+2)
            ax.set_ylabel('Value', fontsize=fontsize)
            ax.grid(True, alpha=0.3)
        
            for spine in ax.spines.values():
                spine.set_alpha(0.3)

        axes[-1].set_xlabel('Time (t)', fontsize=fontsize, labelpad=20)
        unique_handles_labels = dict(zip(labels, handles))
        fig.legend(
            unique_handles_labels.values(), unique_handles_labels.keys(),
            loc='lower center', fontsize=fontsize, frameon=False,
            ncol=(len(unique_handles_labels) // 2 + len(unique_handles_labels) % 2),
            bbox_to_anchor=(0.5, 0)
        )
        
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()


    def create_windows(self, data, window_size):
        """
        Create sliding windows of data.
        """
        windows = []
        for i in range(len(data) - window_size + 1):
            windows.append(data[i:i + window_size])
        return torch.stack(windows)
    

    def preprocess_data(self):
        """
        Get agents data.
        """
        agents_data = {"train": {}, "test": {}}

        for type in agents_data.keys():
            for agent_id, series in self.data[type].items():
                features_num = len(series)
                
                feature_lengths = [len(feature) for feature in series]
                if len(set(feature_lengths)) > 1:
                    raise ValueError("[DATA] All features must have the same length", feature_lengths)
                
                features = []
                for feature in series:
                    feature_tensor = torch.tensor(feature).float()
                    features.append(feature_tensor)
                
                combined_features = torch.stack(features, dim=1)  # (total_length, features_num)
                windows = self.create_windows(combined_features, self.window_size)  # (num_windows, window_size, features_num)
                
                agents_data[type][agent_id] = {
                    "features": features_num,
                    "series": windows
                }
        return agents_data