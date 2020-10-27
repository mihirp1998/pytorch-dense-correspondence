### Updates 

- September 4, 2018: Tutorial and data now available!  [We have a tutorial now available here](./doc/tutorial_getting_started.md), which walks through step-by-step of getting this repo running.
- June 26, 2019: We have updated the repo to pytorch 1.1 and CUDA 10. For code used for the experiments in the paper see [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/releases/tag/pytorch-0.3).


## Dense Correspondence Learning in PyTorch

In this project we learn Dense Object Nets, i.e. dense descriptor networks for previously unseen, potentially deformable objects, and potentially classes of objects:

![](./doc/caterpillar_trim.gif)  |  ![](./doc/shoes_trim.gif) | ![](./doc/hats_trim.gif)
:-------------------------:|:-------------------------:|:-------------------------:

We also demonstrate using Dense Object Nets for robotic manipulation tasks:

![](./doc/caterpillar_grasps.gif)  |  ![](./doc/shoe_tongue_grasps.gif)
:-------------------------:|:-------------------------:

### Dense Object Nets: Learning Dense Visual Descriptors by and for Robotic Manipulation

This is the reference implementation for our paper:

[PDF](https://arxiv.org/pdf/1806.08756.pdf) | [Video](https://www.youtube.com/watch?v=L5UW1VapKNE)

[Pete Florence*](http://www.peteflorence.com/), [Lucas Manuelli*](http://lucasmanuelli.com/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

<em><b>Abstract:</b></em> What is the right object representation for manipulation? We would like robots to visually perceive scenes and learn an understanding of the objects in them that (i) is task-agnostic and can be used as a building block for a variety of manipulation tasks, (ii) is generally applicable to both rigid and non-rigid objects, (iii) takes advantage of the strong priors provided by 3D vision, and (iv) is entirely learned from self-supervision.  This is hard to achieve with previous methods: much recent work in grasping does not extend to grasping specific objects or other tasks, whereas task-specific learning may require many trials to generalize well across object configurations or other tasks.  In this paper we present Dense Object Nets, which build on recent developments in self-supervised dense descriptor learning, as a consistent object representation for visual understanding and manipulation. We demonstrate they can be trained quickly (approximately 20 minutes) for a wide variety of previously unseen and potentially non-rigid objects.  We additionally present novel contributions to enable multi-object descriptor learning, and show that by modifying our training procedure, we can either acquire descriptors which generalize across classes of objects, or descriptors that are distinct for each object instance. Finally, we demonstrate the novel application of learned dense descriptors to robotic manipulation. We demonstrate grasping of specific points on an object across potentially deformed object configurations, and demonstrate using class general descriptors to transfer specific grasps across objects in a class. 

#### Citing

If you find this code useful in your work, please consider citing:

```
@article{florencemanuelli2018dense,
  title={Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation},
  author={Florence, Peter and Manuelli, Lucas and Tedrake, Russ},
  journal={Conference on Robot Learning},
  year={2018}
}
```

### Tutorial

- [getting started with pytorch-dense-correspondence](./doc/tutorial_getting_started.md)

### Code Setup

- [setting up docker image](doc/docker_build_instructions.md)
- [recommended docker workflow ](doc/recommended_workflow.md)

### Dataset

- [data organization](doc/data_organization.md)
- [data pre-processing for a single scene](doc/data_processing_single_scene.md)

### Training and Evaluation
- [training a network](doc/training.md)
- [evaluating a trained network](doc/dcn_evaluation.md)
- [pre-trained models](doc/model_zoo.md)

### Miscellaneous
- [coordinate conventions](doc/coordinate_conventions.md)
- [testing](doc/testing.md)

### Git management

To prevent the repo from growing in size, recommend always "restart and clear outputs" before committing any Jupyter notebooks.  If you'd like to save what your notebook looks like, you can always "download as .html", which is a great way to snapshot the state of that notebook and share.


### Pytorch disco
```
source source.sh
python dense_correspondence/training/training_tutorial.py
```

### Integrating a new pydisco dataset in this repo for training
```
Conda env: denseobj
TB and model dir: /projects/katefgroup/datasets/denseobj_clevr//pdc/code/data_volume/pdc/trained_models/tutorials/clevr_3
```

See this commit on how to integrate a new dataset for training and evaluation:
```https://github.com/mihirp1998/pytorch-dense-correspondence/commit/6b9c5113e661d111b111c3d5b3306f92b9e37330```

1. Execute ```test/clevr_segmentation_mask_gen.py``` script in qnet_corres_entity branch or create similar script for your dataset

2. Execute ```test/clevr_denseobjnet_dataset_creation.py``` script in qnet_corres_entity branch or create similar script for your dataset. Change stuff with 'change this for new dataset' written in 
front of it.

3. Change ```config/params.py```

4. Change intrinsics in ```get_default_K_matrix```

5. Change mean and variance in ```constants.py```

6. Create config files like carla_4cars.yaml and carla_4cars_train.yaml. Run ```pickle_names_yaml_creator.py``` which will give you pickle files names which you can copy paste in _train.yaml file

7. Change config_filename in ```training_tutorial.py```. Change ```self.train_config["training"]["logging_dir_name"]``` there too.

8. Set ```descr_dim``` as required in training_tutorial.py

9. Change image H and W in training.yaml

10. Change default pickle folder path in ```get_rgbd_mask_pose``` in ```dense_correspondence_dataset_masked.py```

11. Change image_height and image_width in DenseCorrespondenceNetwork

12. If you get the pickle protocol=3 error, run ```pickle_protocol_changer.py script``` to chane pickle 
files to protocol 2.


### Integrating a new pydisco dataset in this repo for evaluation

1. Change model_path and pickle_folder in evaluation.yaml

2. Add model name in heatmap.yaml in networks and change pickle_folder there too

3. Change model yaml in annotate_correspondence.py and live_heatmap_visualization.py


### Live heatmap visualization

ssh -Y mprabud@matrix.ml.cmu.edu 

DON'T GO IN ANY SCREEN

then ssh -Y compute-0-38

then:

cd /home/mprabhud/projects/pytorch-dense-correspondence

source source.sh

python modules/user-interaction-heatmap-visualization/live_heatmap_visualization.py

then u can see the visualization on ur mac

via the cluster

ssh -Y is transfering the display ports to ur local pc
