import os
import argparse
import tensorflow as tf

from process_data import ProcessData
from data_io import IO
from model import grainedDKTModel, BatchGenerator, run

DATA_FOLDER = '../data/'
ASSISTment_Datafile = 'skill_builder_data.csv'
ASSISTment_train = '../data/training_ASSISTment_all.csv'
ASSISTment_test = '../data/testing_ASSISTment_all.csv'
ASSISTment_evaluate_result = 'ASSISTment_results.csv'
ASSISTment_Category_MappingFile = '../data/skill-category(UTF-8).csv'
MODEL_LOG_FOLDER = "../logs/"
MODEL_FOLDER = "../model/"
MODEL_FILE = 'model.ckpt'

PKU_Datafile = '../data/PKU_MOOC/question_sessions.csv'
PKU_train = '../data/PKU_MOOC/training.csv'
PKU_test = '../data/PKU_MOOC/testing.csv'
PKU_evaluate_result = '../data/PKU_MOOC/PKU_results.csv'
PKU_Category_MappingFile = '../data/PKU_MOOC/question_category.csv'

fname_TrainData = PKU_train
fname_TestData = PKU_test
fname_MapData = PKU_Category_MappingFile
fname_Result = PKU_evaluate_result

DATA_READY = True
SILENT_WARNINGS = True # to silent the warnings (https://github.com/tensorflow/tensorflow/issues/8037)

if not DATA_READY:
    PrePrcess = ProcessData(data_folder = DATA_FOLDER)
    PreProcess.ASSISTment_load_save(ASSISTment_Datafile)
if SILENT_WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main(args):
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    n_step = args.n_step
    keep_prob = args.keep_prob
    n_hidden_units = args.n_hidden_units
    embedding_size = args.embedding_size
    initial_learning_rate=args.initial_learning_rate
    final_learning_rate=args.final_learning_rate
    assert args.embedding in ['random', 'one_hot'], 'Unrecognized embedding method'
    random_embedding = True if args.embedding == 'random' else False
    assert args.granularity in ['single', 'multi'], 'Unrecognized granularity'
    multi_granined = True if args.granularity == 'multi' else False
    assert args.granularity_out in ['single', 'multi'], 'Unrecognized granularity'
    multi_granined_out = True if args.granularity_out == 'multi' else False
    assert args.train_mode in ['step', 'epoch'], 'Unreconized traning mode (please specify step or epoch)'
    
    PrepData = IO()
    train_response_list, question_list = PrepData.load_model_input(fname_TrainData, sep=',')
    test_response_list, question_list = PrepData.load_model_input(fname_TestData, sep=',', question_list=question_list)
    id_encoding = PrepData.question_id_1hotencoding(question_list)
    category_map_dict = PrepData.load_category_map(fname_MapData, sep=',')
    category_encoding = PrepData.category_id_1hotencoding(category_map_dict)

    skill2category_map =PrepData.skill_idx_2_category_idx(category_map_dict, category_encoding)
    n_id = len(id_encoding)
    n_categories = len(category_encoding)

    # print {skill: category_encoding[category_map_dict[skill]] for skill in category_map_dict.keys()}
    # print skill2category_map

    train_batches = BatchGenerator(train_response_list, batch_size, id_encoding, n_id, n_id, n_categories, random_embedding=random_embedding, skill_to_category_dict=skill2category_map, multi_granined_out=multi_granined_out)
    test_batches = BatchGenerator(test_response_list, batch_size, id_encoding, n_id, n_id, n_categories, random_embedding=random_embedding, skill_to_category_dict=skill2category_map, multi_granined_out=multi_granined_out)

    sess = tf.Session()
    run(sess, train_batches, test_batches, \
        option=args.train_mode, record_performance=True, \
        model_saved_path=os.path.join(MODEL_FOLDER, MODEL_FILE),
        n_step=n_step, random_embedding=random_embedding, multi_granined=multi_granined, \
        n_categories=n_categories, out_folder=DATA_FOLDER, out_file=fname_Result, \
        keep_prob=keep_prob, n_hidden_units=n_hidden_units, embedding_size=embedding_size, \
        initial_learning_rate=0.001, final_learning_rate=0.00001,
        multi_granined_out=multi_granined_out)
    # tensorboard --logdir logs
    writer = tf.summary.FileWriter(MODEL_LOG_FOLDER, sess.graph) # http://localhost:6006/#graphs on mac
    sess.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',     
                        dest='batch_size',
                        # required=True,
                        default=16,
                        type=int,
                        help="Batch size")
    parser.add_argument('--train-steps',
                        dest='n_step',
                        # required=True,
                        default=5001,
                        type=int,
                        help='Maximum number of training steps to perform.')
    parser.add_argument('--train-epochs',
                        dest='n_epoch',
                        # required=True,
                        default=20,
                        type=int,
                        help='Maximum number of training epochs to perform.')
    parser.add_argument('--train-mode',     
                        dest='train_mode',
                        default="step",
                        type=str,
                        help="By Step or By Epoch")
    parser.add_argument('--num-hiddenunits',     
                        dest='n_hidden_units',
                        default=200,
                        type=int,
                        help="Number of hidden units")
    parser.add_argument('--embedding-size',     
                        dest='embedding_size',
                        default=200,
                        type=int,
                        help="Size of embedded input vectors")
    parser.add_argument('--droupout-keep',     
                        dest='keep_prob',
                        default=0.5,
                        type=float,
                        help="Size of embedded input vectors")
    parser.add_argument('--learningrate-init',     
                        dest='initial_learning_rate',
                        default=0.001,
                        type=float,
                        help="Initialized learning rate")
    parser.add_argument('--learningrate-final',     
                        dest='final_learning_rate',
                        default=0.00001,
                        type=float,
                        help="Final learning rate")
    parser.add_argument('--input-embedding',     
                        dest='embedding',
                        default="random",
                        type=str,
                        help="Use randomized embedding or 1-hot embedding as inputs?")
    parser.add_argument('--input-grain',     
                        dest='granularity',
                        default="single",
                        type=str,
                        help="Use standard information or multi-grained information?")
    parser.add_argument('--output-grain',     
                        dest='granularity_out',
                        default="single",
                        type=str,
                        help="Use standard information or multi-grained information for output?")
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity('INFO')
    args = parser.parse_args()
    main(args)