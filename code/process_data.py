import pandas as pd
import numpy as np
import os
import random
import math

class ProcessData:
    def __init__(self,limit=None, data_folder='./', max_session_len=60000):
        self.limit = limit
        self.data_folder = data_folder
        self.max_session_len = max_session_len
    def oneD_array2str(self, a, sep=','):
        return sep.join(str(elem) for elem in a)
    def ASSISTment_load_save(self, filename, sep=',', outfilename='ASSISTment_all.csv'):
        filepath = os.path.join(self.data_folder, filename)
        filepath_out = os.path.join(self.data_folder, outfilename)
        filepath_out_training = os.path.join(self.data_folder, "training_" + outfilename)
        filepath_out_testing = os.path.join(self.data_folder, "testing_" + outfilename)
        df = pd.read_csv(filepath, sep=sep)
        n_records = len(df) if self.limit == None else self.limit
        n_slots = n_records // self.max_session_len
        with open(filepath_out, 'a') as out_file:
            print("successfully opened file {0} for writing".format(filepath_out))
            for slot_idx in range(n_slots):
                start_idx = slot_idx * self.max_session_len
                end_idx = min((slot_idx + 1) * self.max_session_len, n_records)
                slot_length = end_idx - start_idx
                visited = np.zeros(slot_length)
                current_slot = df.iloc[start_idx: end_idx]
                n_records_current = len(current_slot)
                user_list = set(current_slot['user_id'])
                print("dealing with slot {0}/{1}, length {2}, {3} users".format(slot_idx, n_slots, n_records_current, len(user_list)))
                grouped = current_slot.groupby(['user_id'])
                # print (grouped.size())
                user_idx = 0
                for user_id, group in grouped:
                    skills_buff = []
                    correct_buff = []
                    n_count = len(group)
                    user_idx += 1
                    print("scanning user {0}/{1} ({4} records) in slot {2}/{3}".format(user_idx, len(user_list), (slot_idx+1), n_slots, n_count))
                    if n_count > 1:
                        for line_idx in range(n_count):
                            current_line = group.iloc[line_idx]
                            # current_user = current_line['user_id']
                            skill_id = current_line['skill_id']
                            skill_name = current_line['skill_name']
                            correct = current_line['correct']
                            if math.isnan(skill_id) or math.isnan(correct) or not (isinstance(skill_name, basestring) and len(skill_name) > 0):
                                continue
                            current_skill = int(skill_id)
                            current_correct = int(correct)
                            skills_buff.append(current_skill)
                            correct_buff.append(current_correct)
                        n_count_valid = len(skills_buff)
                        if n_count_valid > 1:
                            out_file.write("{0}\n".format(n_count_valid))
                            out_file.write("{0}\n".format(self.oneD_array2str(skills_buff)))
                            out_file.write("{0}\n".format(self.oneD_array2str(correct_buff)))
                            if n_count_valid > 30 or len(set(skills_buff)) > 3:
                                filepath_out_selected = [ 
                                        filepath_out_testing,
                                        filepath_out_training,
                                        filepath_out_training,
                                        filepath_out_training,
                                        filepath_out_training
                                    ]
                                with open(random.choice(filepath_out_selected), 'a') as out_selected:
                                    out_selected.write("{0}\n".format(n_count_valid))
                                    out_selected.write("{0}\n".format(self.oneD_array2str(skills_buff)))
                                    out_selected.write("{0}\n".format(self.oneD_array2str(correct_buff)))
