from process_data import ProcessData

DATA_FOLDER = '../data/'
ASSISTment_Datafile = 'skill_builder_data.csv'

DATA_READY = False

DataPrep = ProcessData(data_folder = DATA_FOLDER)
if not DATA_READY:
    DataPrep.ASSISTment_load_save(ASSISTment_Datafile)