import socket
hostname = socket.gethostname()
if "compute" in hostname:
	DIR_DATA = '/projects/katefgroup/datasets/denseobj_carla/'
	DIR_PROJ = '/home/mprabhud/projects/pytorch-dense-correspondence'
else:
	DIR_DATA = '/media/mihir/dataset/denseobjnet_carla'
	DIR_PROJ = '/home/mihir/Documents/projects/pytorch-dense-correspondence'