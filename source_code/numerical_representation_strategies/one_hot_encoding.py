from numerical_representation_strategies.constant_values import constant_values
from joblib import Parallel, delayed
import pandas as pd

class one_hot_encoding(object):

    def __init__(self, dataset=None, column_response=None, column_seq=None):

        self.dataset = dataset
        self.column_response = column_response
        self.column_seq = column_seq
        self.constant_values_instance = constant_values()

        self.zero_padding = self.check_max_size()

    def __check_residues(self, residue):
        if residue in self.constant_instance.possible_residues:
            return True
        else:
            return False

    def __encode_residue(self, residue):
        vector = [0 for k in range(20)]
        position = self.constant_values_instance.dict_value[residue]
        vector[position] = 1
        return vector

    def __check_max_size(self):
        size_list = [len(seq) for seq in self.dataset[self.name_column_seq]]

        max_element = max(size_list)*20

        return max_element
        
    def __encoding_sequence(self, sequence, id_seq):

        sequence = sequence.upper()
        sequence_encoding = []

        for i in range(len(sequence)):
            residue = sequence[i]
            if self.__check_residues(residue):
                response_encoding = self.__encoding_residue(residue)
                sequence_encoding = sequence_encoding + response_encoding

        # complete zero padding
        for k in range(len(sequence_encoding), self.zero_padding):
            sequence_encoding.append(0)

        sequence_encoding.insert(0, id_seq)
        return sequence_encoding
    
    def encoding_dataset(self):

        #print("Start encoding process")
        data_encoding = Parallel(n_jobs=self.constant_instance.n_cores, require='sharedmem')(delayed(self.__encoding_sequence)(self.dataset[self.name_column_seq][i], self.dataset[self.name_column_id][i]) for i in range(len(self.dataset)))

        print("Processing results")
        matrix_data = []
        for element in data_encoding:
            matrix_data.append(element)

        print("Creating dataset")
        header = ['p_{}'.format(i) for i in range(len(matrix_data[0])-1)]
        header.insert(0, self.name_column_id)
        print("Export dataset")
        df_data = pd.DataFrame(matrix_data, columns=header)


        return df_data