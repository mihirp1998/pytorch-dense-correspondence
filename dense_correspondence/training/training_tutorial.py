import os
import sys
from config.params import *
import logging
os.chdir(DIR_PROJ)
sys.path.append(os.path.join(DIR_PROJ, 'modules'))
os.environ['DC_SOURCE_DIR'] = DIR_PROJ
os.environ['DC_DATA_DIR'] = "{}/pdc".format(DIR_DATA)


# utils.add_dense_correspondence_to_python_path()

import modules.dense_correspondence_manipulation.utils.utils as utils
from dense_correspondence.training.training import *
from dense_correspondence.training.training import DenseCorrespondenceTraining
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset
from dense_correspondence.evaluation.evaluation import DenseCorrespondenceEvaluation
import ipdb
st = ipdb.set_trace
class TrainingTutorial:
    def __init__(self):

        logging.basicConfig(level=logging.INFO)

        self.load_configuration()


    def load_configuration(self):
        # config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
        config_filename = os.path.join(DIR_PROJ, 'config', 'dense_correspondence',
                                       'dataset', 'composite', 'carla_4cars.yaml')
        config = utils.getDictFromYamlFilename(config_filename)
        # train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
        train_config_file = os.path.join(DIR_PROJ, 'config', 'dense_correspondence',
                                         'training', 'training.yaml')
        self.train_config = utils.getDictFromYamlFilename(train_config_file)
        self.dataset = SpartanDataset(config=config)
        # st()

        logging_dir = "code/data_volume/pdc/trained_models/tutorials"
        num_iterations = 3500
        descr_dim = 3  # the descriptor dimension
        self.train_config["training"]["logging_dir_name"] = "carla_%d" % (descr_dim)
        self.train_config["training"]["logging_dir"] = logging_dir
        self.train_config["dense_correspondence_network"]["descriptor_dimension"] = descr_dim
        self.train_config["training"]["num_iterations"] = num_iterations


    def train(self):
        # This should take about ~12-15 minutes with a GTX 1080 Ti

        # All of the saved data for this network will be located in the
        # code/data_volume/pdc/trained_models/tutorials/caterpillar_3 folder

        descr_dim = self.train_config["dense_correspondence_network"]["descriptor_dimension"]
        print("training descriptor of dimension %d" % (descr_dim))
        train = DenseCorrespondenceTraining(dataset=self.dataset, config=self.train_config)
        train.run()
        print("finished training descriptor of dimension %d" % (descr_dim))


    def evaluate(self):
        logging_dir = self.train_config["training"]["logging_dir"]
        logging_dir_name = self.train_config["training"]["logging_dir_name"]
        model_folder = os.path.join(logging_dir, logging_dir_name)
        model_folder = utils.convert_to_absolute_path(model_folder)

        DCE = DenseCorrespondenceEvaluation
        num_image_pairs = 100
        DCE.run_evaluation_on_network(model_folder, num_image_pairs=num_image_pairs)

if __name__ == "__main__":
    trainTut = TrainingTutorial()
    trainTut.train()
    print("hello")