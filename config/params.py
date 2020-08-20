import socket
hostname = socket.gethostname()
if "compute" in hostname:
	DIR_DATA = '/projects/katefgroup/datasets/denseobj/'
	DIR_PROJ = '/home/mprabhud/projects/pytorch-dense-correspondence'
else:
	DIR_DATA = '/media/mihir/dataset/denseobjnet'
	DIR_PROJ = '/home/mihir/Documents/projects/pytorch-dense-correspondence'