import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ObjetiveFunctionAggregationsHeatmap(object):

    def __init__(self,
                 files_dir:str, 
                 save_dir:str, 
                 number_of_executions:int, 
                 base_file:str, 
                 file_format:str='csv') -> None:

        self._files_dir = files_dir
        self._save_dir = save_dir
        self._number_of_executions = number_of_executions
        self._base_file = base_file
        self._file_format = file_format
        self._create_heat_map()
    
    def _create_heat_map(self) -> None:
        aggregations = []
        for i in range(self._number_of_executions):
            file_name = self._base_file + f'-{i}.' + self._file_format
            aggregations.append(pd.read_csv(os.path.join(self._files_dir, file_name.format(i)), header=None))

        aggregations = pd.concat(aggregations, axis=1)
        aggregations.columns = ['exec_{}'.format(i) for i in range(self._number_of_executions)]
        aggregation_list = [aggregations['exec_{}'.format(i)].value_counts().keys().values.tolist() 
                            for i in range(self._number_of_executions)]
        unique_aggregations = list({[j  for i in aggregation_list for j in i]})
        unique_aggregations.sort(key = lambda x: x.split('-')[1], reverse=True)
        data_transposed = aggregations.T

        number_of_aggregations = len(unique_aggregations)
        number_of_generations = len(aggregations.index)
        self._heat_map = pd.DataFrame(data=np.zeros((number_of_aggregations, number_of_generations)))
        self._heat_map.index = unique_aggregations

        for i in range(number_of_generations):
            for k,v in data_transposed[i].value_counts().items():
                self._heat_map.at[k, i] = v

        for i in range(number_of_generations):
            if self._heat_map[i].values.sum() != self._number_of_executions:
                print('Error in generation {}'.format(i))

    def show_heat_map(self) -> None:
        plt.figure(figsize=(18,10))
        sns.heatmap(self._heat_map.values, yticklabels=self._heat_map.index.values, cmap="Blues")

        plt.xlabel('Generation', fontsize=20)

        plt.yticks(fontsize=13)
        plt.ylabel('Aggregation', fontsize=20)

        plt.title('Aggregation Heat Map', fontsize=20)
        plt.savefig(os.path.join(self._save_dir, 'heat_map.pdf'))
        plt.show()