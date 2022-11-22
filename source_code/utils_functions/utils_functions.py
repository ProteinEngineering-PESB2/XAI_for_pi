from Bio.PDB import *
from Bio import SeqIO
import pandas as pd
from joblib import dump
from scipy.spatial import distance

class utils_functions(object):

    def get_distance_vectors(self, vector1, vector2, type_distance):

        if type_distance == 1:
            return distance.euclidean(vector1, vector2)
        elif type_distance == 2:
            return distance.braycurtis(vector1, vector2)
        elif type_distance == 3:
            return distance.canberra(vector1, vector2)
        elif type_distance == 4:
            return distance.chebyshev(vector1, vector2)
        elif type_distance == 5:
            return distance.cityblock(vector1, vector2)
        elif type_distance == 6:
            return distance.correlation(vector1, vector2)
        elif type_distance == 7:
            return distance.cosine(vector1, vector2)
        elif type_distance == 8:
            return distance.minkowski(vector1, vector2)
        else:
            return distance.hamming(vector1, vector2)

    def export_csv(self, dataset, name_doc):

        dataset.to_csv(name_doc, index=False)

    def pdb_parser(self, code_pdb, name_pdb, type_document):

        if type_document == 'PDB':
            parser = PDBParser()
        else:
            parser = MMCIFParser()

        structure = parser.get_structure(code_pdb, name_pdb)
        return structure

    def numpy_array_to_df(self, array_data, column_response, response, is_export, name_export=None):

        header = ["p_{}".format(i) for i in range(len(array_data[0]))]
        df_data = pd.DataFrame(array_data, columns=header)
        df_data[column_response] = response

        if is_export:
            self.export_csv(df_data, name_export)

        return df_data

    def export_instance(self, instance_export, name_export):

        dump(instance_export, name_export)

    def csv_to_fasta(self, dataset, column_seq, column_response, name_export):

        print("Start export csv to fasta")
        try:
            doc_export = open(name_export, 'w')

            for i in range(len(dataset)):
                seq = dataset[column_seq][i]
                response = dataset[column_response][i]

                text = ">seq {} | response {}\n{}".format(i, response, seq)
                if i != len(dataset) - 1:
                    text += "\n"

                doc_export.write(text)
            print("End create file")
            doc_export.close()
            return True
        except:
            return False

    def fasta_to_csv(self, document_to_process, separator_response):

        matrix_data = []
        with open(document_to_process) as handle:
            for record in SeqIO.parse(handle, "fasta"):

                response = record.description.split(separator_response)[-1].strip()
                row = [record.id, response, str(record.seq)]
                matrix_data.append(row)
        df_data = pd.DataFrame(matrix_data, columns=['id', 'response', 'seq'])
        return df_data

    def run_external_command(self, command, path, name_export):

        import os
        print(command)
        os.system(command)

        list_files = os.listdir(path)
        print(list_files)
        print(name_export)
        if name_export in list_files:
            print("Is OK")
            return True
        else:
            return False