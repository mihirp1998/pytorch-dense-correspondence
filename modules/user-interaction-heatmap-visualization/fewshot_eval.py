import sys
import os
import cv2
import numpy as np
import copy
from config.params import *
import logging
import pickle
os.environ['DC_SOURCE_DIR'] = DIR_PROJ
os.environ['DC_DATA_DIR'] = "{}/pdc".format(DIR_DATA)
import ipdb
st = ipdb.set_trace
# st()
from sklearn.decomposition import PCA
import dense_correspondence_manipulation.utils.utils as utils

dc_source_dir = utils.getDenseCorrespondenceSourceDir()
sys.path.append(dc_source_dir)
sys.path.append(os.path.join(dc_source_dir, "dense_correspondence", "correspondence_tools"))
from dense_correspondence.dataset.spartan_dataset_masked import SpartanDataset, ImageType

import matplotlib.pyplot as plt
import dense_correspondence
from dense_correspondence.evaluation.evaluation import *
from dense_correspondence.evaluation.plotting import normalize_descriptor
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork


import dense_correspondence_manipulation.utils.visualization as vis_utils


from dense_correspondence_manipulation.simple_pixel_correspondence_labeler.annotate_correspondences import label_colors, draw_reticle, pil_image_to_cv2, drawing_scale_config, numpy_to_cv2




COLOR_RED = np.array([0, 0, 255])
COLOR_GREEN = np.array([0,255,0])

utils.set_default_cuda_visible_devices()
eval_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 'evaluation', 'evaluation.yaml')
EVAL_CONFIG = utils.getDictFromYamlFilename(eval_config_filename)



LOAD_SPECIFIC_DATASET = True

class FewShotEvaluation(object):
    """
    For few show style content evaluation

    Edit config/dense_correspondence/heatmap_vis/heatmap.yaml to specify which networks
    to visualize. Specifically add the network you want to visualize to the "networks" list.
    Make sure that this network appears in the file pointed to by EVAL_CONFIG

    Usage: Launch this file with python after sourcing the environment with
    `use_pytorch_dense_correspondence`

    Then `python fewshot_eval.py`.

    Keypresses:
        n: new set of images
        s: swap images
        p: pause/un-pause
    """

    def __init__(self, config):
        self.shot = 5 #1 or 5
        self.do_fewshot_style = False
        if self.do_fewshot_style:
            self.do_fewshot_content = False
        else:
            self.do_fewshot_content = True

        self.total = 0
        self.correct = 0
        self.summarize_fewshot_data = True
        self.few_shot_dict = {}

        self._config = config
        self._dce = DenseCorrespondenceEvaluation(EVAL_CONFIG)
        self._load_networks()
        self._reticle_color = COLOR_GREEN
        self._paused = False
        self.hardcode_samples_idx = 0
        self.first_sample_call = 1
        if LOAD_SPECIFIC_DATASET:
            self.load_specific_dataset() # uncomment if you want to load a specific dataset
    
    def is_fewshot_dict_full(self, dictt):
        if self.summarize_fewshot_data == False:
            return True
        total = 3
        if self.do_fewshot_style:
            total = 16
        if len(dictt.keys()) < total:
            print("Keys not enough, {}/{} keys".format(len(dictt.keys()), total))
            return False 
        for key in dictt.keys():
            if len(dictt[key]) < self.shot:
                print("Less enteries-- {}:{}".format(key, len(dictt[key])))
                return False 
        return True

    def _load_networks(self):
        # we will use the dataset for the first network in the series
        self._dcn_dict = dict()

        self._dataset = None
        self._network_reticle_color = dict()
        for idx, network_name in enumerate(self._config["networks"]):
            dcn = self._dce.load_network_from_config(network_name)
            dcn.eval()
            self._dcn_dict[network_name] = dcn
            # self._network_reticle_color[network_name] = label_colors[idx]

            if len(self._config["networks"]) == 1:
                self._network_reticle_color[network_name] = COLOR_RED
            else:
                self._network_reticle_color[network_name] = label_colors[idx]

            if self._dataset is None:
                self._dataset = dcn.load_training_dataset()

    def load_specific_dataset(self):
        dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                            'dataset', 'composite', 'clevr_single_large.yaml')

        # dataset_config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config',
        #                                        'dense_correspondence',
        #                                        'dataset', 'composite', '4_shoes_all.yaml')
        # st()
        dataset_config = utils.getDictFromYamlFilename(dataset_config_filename)
        self._dataset = SpartanDataset(config=dataset_config)

    def get_random_image_pair(self):
        """
        Gets a pair of random images for different scenes of the same object
        """
        # st()
        object_id = self._dataset.get_random_object_id()
        # scene_name_a = "2018-04-10-16-02-59"
        # scene_name_b = scene_name_a

        scene_name_a = self._dataset.get_random_single_object_scene_name(object_id)
        # scene_name_b = self._dataset.get_different_scene_for_object(object_id, scene_name_a)

        image_a_idx = self._dataset.get_random_image_index(scene_name_a)
        # image_b_idx = self._dataset.get_random_image_index(scene_name_b)

        return scene_name_a, image_a_idx


    def _get_new_images(self):
        """
        Gets a new pair of images
        :return:
        :rtype:
        """
        if random.random() < 0.5:
            self._dataset.set_train_mode()
        else:
            self._dataset.set_test_mode()

        scene_name_1, image_1_idx = self.get_random_image_pair()

        self.img1_pil = self._dataset.get_rgb_image_from_scene_name_and_idx(scene_name_1, image_1_idx)

        self._scene_name_1 = scene_name_1
        self._image_1_idx = image_1_idx
        self.pickle_path = self._config['pickle_folder'][0]

        print("scene1: {}, idx1: {}".format(self._scene_name_1, self._image_1_idx))

        self._compute_descriptors()


    def _compute_descriptors(self):
        """
        Computes the descriptors for image 1 and image 2 for each network
        :return:
        :rtype:
        """
        self.img1 = pil_image_to_cv2(self.img1_pil)
        self.rgb_1_tensor = self._dataset.rgb_image_to_tensor(self.img1_pil)

        self._res_a = dict()
        for network_name, dcn in self._dcn_dict.iteritems():
            self._res_a[network_name] = dcn.forward_single_image_tensor(self.rgb_1_tensor).data.cpu().numpy()
            self._res_a[network_name] = self._res_a[network_name].reshape(-1, self._res_a[network_name].shape[-1])

        fname = os.path.join(self.pickle_path, self._scene_name_1 + '.p')
        pfile1 = pickle.load(open(fname, 'rb'))
        segment1 = pfile1['segment_camXs_raw'][self._image_1_idx].transpose(1,2,0)
        segment1 = segment1.reshape(-1)
        occupied_idxs = np.where(segment1 == 1)[0]
        occupied_idxs_feats = np.mean(self._res_a[network_name][occupied_idxs], axis=0)
        if self.do_fewshot_style:
            key = pfile1['material_list'][0]+ '_' + pfile1['color_list'][0]
        else:
            key = pfile1['shape_list'][0]
        key_orig = key

        if self.is_fewshot_dict_full(self.few_shot_dict):
            if self.summarize_fewshot_data:
                self.summarize_fewshot_data = False
                for key in self.few_shot_dict.keys():
                    value = self.few_shot_dict[key]
                    value = np.stack(value)
                    self.few_shot_dict[key] = np.mean(value, axis=0)
            
            maxi = -100000000
            maxi_label = ""
            for key in self.few_shot_dict.keys():
                value = self.few_shot_dict[key]
                dotprod = np.sum(value*occupied_idxs_feats)/(np.linalg.norm(value)*np.linalg.norm(occupied_idxs_feats) + 1e-5)
                if dotprod > maxi:
                    maxi = dotprod
                    maxi_label = key

            if maxi_label  == key_orig:
                self.correct += 1.0
            self.total += 1.0
            print("Accuracy : {}".format(self.correct/self.total))
        
        else:
            print("inserting key: ", key)
            if self.should_insert_key(key):
                self.few_shot_dict[key].append(occupied_idxs_feats)
            return None

            
    
    def should_insert_key(self, key):
        if key not in self.few_shot_dict:
            self.few_shot_dict[key] = []
            return True
        if len(self.few_shot_dict[key])<self.shot:
            return True
        return False


    def run(self):
        while True:
            self._get_new_images()
        

if __name__ == "__main__":
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()
    config_file = os.path.join(dc_source_dir, 'config', 'dense_correspondence', 'heatmap_vis', 'heatmap.yaml')
    config = utils.getDictFromYamlFilename(config_file)

    heatmap_vis = FewShotEvaluation(config)
    print "starting heatmap vis"
    heatmap_vis.run()
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
