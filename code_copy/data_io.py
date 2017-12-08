import csv
import pandas as pd

class IO:
    class CSVReader:
        def __init__(self, filename, delimiter='\t'):
            self.csvfile = open(filename, "r")
            self.csvreader = csv.reader(self.csvfile, delimiter=delimiter)
        def __del__(self):
            self.csvfile.close()
        def read_next_line(self):
            next_line = None
            while not next_line:
                try:
                    next_line = next(self.csvreader)
                except StopIteration:
                    raise StopIteration
                    break
            return next_line
    def load_model_input(self, filename, question_list=[], sep='\t'):
        # question_list = []
        response_list = []
        csvreader = self.CSVReader(filename, delimiter=sep)
        while True:
            try:
                seq_length_line = csvreader.read_next_line()
                seq_questionsID = csvreader.read_next_line()
                seq_correctness = csvreader.read_next_line()
                seq_length = int(seq_length_line[0])
                assert len(seq_length_line) == 1 and seq_length == len(seq_questionsID) and seq_length == len(seq_correctness), \
                    "Unexpected format of input CSV file in {0}\n:{1}\n{2}\n{3}".format(filename, seq_length_line, seq_questionsID, seq_correctness)
                if seq_length > 1: # only when there are at least two questions together is the sequence meaningful
                    question_list += [question for question in set(seq_questionsID) if question not in question_list]
                    response_list.append((seq_length, list(zip(map(int, seq_questionsID), map(int, seq_correctness)))))
            except StopIteration:
                print ("reached the end of the file {0}".format(filename))
                break
        del csvreader
        return response_list, question_list
    def question_id_1hotencoding(self, question_list):
        id_encoding = { int(j): int(i) for i, j in enumerate(question_list)}
        return id_encoding
    def load_category_map(self, filename, sep='\t'):
        category_map_dict = {}
        mapping_csv = pd.read_csv(filename, sep=sep)
        sum_skill_num = len(mapping_csv)
        for idx in range(sum_skill_num):
            skill_id = mapping_csv.iloc[idx]['skill_id']
            category_id = mapping_csv.iloc[idx]['category_id']
            category_map_dict[skill_id] = category_id
        return category_map_dict
    def category_id_1hotencoding(self, skill_to_category_dict):
        categories_list = set(skill_to_category_dict.values())
        category_encoding = { int(j): int(i) for i, j in enumerate(categories_list)}
        return category_encoding
    def skill_idx_2_category_idx(self, category_map_dict, category_encoding):
        return {skill: category_encoding[category_map_dict[skill]] for skill in category_map_dict.keys()}


        