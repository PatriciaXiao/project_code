from datetime import datetime
import pandas as pd
import random

# Mon  2 Dec 2013  3:44 AM UTC
# datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
# datetime.strptime('Mon  2 Dec 2013  3:44 AM UTC', '%a  %d %b %Y  %I:%M %p UTC')
def time_parser(time_str, time_format='%a  %d %b %Y  %I:%M %p UTC'):
    time_obj = datetime.strptime(time_str, time_format)
    detailed_time = (time_obj.year, time_obj.month, time_obj.day, time_obj.hour, time_obj.minute)
    rough_time = (time_obj.year, time_obj.month)
    return detailed_time, rough_time

def oneD_array2str(a, sep=','):
    return sep.join(str(elem) for elem in a)

QUIZ_ID = {
    '1-1': '43', 
    '1-2': '45', 
    '2-1': '57', 
    '2-2': '59', 
    '3': '63', 
    '4': '79', 
    '5': '109', 
    '6': '111', 
    '7': '113',
    '8': '125',
    '9': '143',
    '10': '145',
    '11': '159',
    '12': '175',
}


list_user = []
list_qst = []
list_ans = []
list_time = []
list_group = []
category_data = '../data/PKU_MOOC/question_category.csv'
session_data = '../data/PKU_MOOC/question_sessions.csv'
filepath_out_training = '../data/PKU_MOOC/training.csv'
filepath_out_testing = '../data/PKU_MOOC/testing.csv'
with open(category_data, 'w') as f:
    f.write('skill_id,category_id\n')
with open(filepath_out_training, 'w') as f:
    f.write('')
with open(filepath_out_testing, 'w') as f:
    f.write('')
sep='\t'
questionID = 1
# original format like
# session_user_id   coursera_user_id    submission_id   A   B   C_Checkbox  E   F   submission_time
for filename in sorted([int(v) for v in QUIZ_ID.values()]):
    filepath = '../data/PKU_MOOC/response/quiz{0}.txt'.format(filename)
    print "loading {0}".format(filepath)
    df = pd.read_csv(filepath, sep=sep, header=None)
    # print df.columns
    n_records = len(df)
    line_end = df.iloc[0][len(df.iloc[0])-1]
    if (isinstance(line_end, basestring) and len(line_end) > 0):
        end_legal = 1
    else:
        end_legal = 2
    n_questions = len(df.iloc[0][3:-1*end_legal])
    with open(category_data, 'a') as f:
        for idx_q in range(n_questions):
            f.write('{0},{1}\n'.format(questionID + idx_q, filename))
    for idx in range(n_records):
        tmp_line = df.iloc[idx]
        user_id = tmp_line[1]
        correct = tmp_line[3:-1*end_legal]
        tmptime, tmptime_rough = time_parser(tmp_line[len(tmp_line)-1*end_legal])
        for idx_q in range(n_questions):
            list_user.append(user_id)
            list_qst.append(questionID + idx_q)
            list_ans.append(correct[3 + idx_q])
            list_time.append(tmptime)
            list_group.append(tmptime_rough)
    questionID += n_questions

dataset_all = pd.DataFrame(data={'user_id': list_user, 'timestamp': list_time, 'group':list_group, 'skill_id': list_qst, 'correct': list_ans})
# sorted_dataset = dataset_all.sort_values(['timestamp'])

data_grouped = dataset_all.groupby(['group'])
for date, group in data_grouped:
    # print date, len(group)
    sorted_group = group.sort_values(['timestamp', 'user_id'])
    # print sorted_group
    n_line = len(sorted_group)
    current_questions = []
    current_corrects = []
    current_user = ''
    for idx in range(n_line):
        tmp_line = sorted_group.iloc[idx]
        user_id = tmp_line['user_id']
        question = tmp_line['skill_id']
        correct = tmp_line['correct']
        if user_id == current_user:
            current_questions.append(question)
            current_corrects.append(correct)
        else:
            len_sess = len(current_questions)
            if len_sess > 1:
                with open(session_data, 'a') as f:
                    f.write("{0}\n{1}\n{2}\n".format(len_sess, oneD_array2str(current_questions), oneD_array2str(current_corrects)))
                filepath_out_selected = [ 
                        filepath_out_testing,
                        filepath_out_training,
                        filepath_out_training,
                        filepath_out_training,
                        filepath_out_training
                    ]
                with open(random.choice(filepath_out_selected), 'a') as out_selected:
                    out_selected.write("{0}\n{1}\n{2}\n".format(len_sess, oneD_array2str(current_questions), oneD_array2str(current_corrects)))
            current_user = user_id
            current_questions = [question]
            current_corrects = [correct]
    len_sess = len(current_questions)
    if len_sess > 1:
        with open(session_data, 'a') as f:
            f.write("{0}\n{1}\n{2}\n".format(len_sess, oneD_array2str(current_questions), oneD_array2str(current_corrects)))
        filepath_out_selected = [ 
                filepath_out_testing,
                filepath_out_training,
                filepath_out_training,
                filepath_out_training,
                filepath_out_training
            ]
        with open(random.choice(filepath_out_selected), 'a') as out_selected:
            out_selected.write("{0}\n{1}\n{2}\n".format(len_sess, oneD_array2str(current_questions), oneD_array2str(current_corrects)))



