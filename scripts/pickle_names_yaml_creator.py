import os
import ipdb 
st = ipdb.set_trace
path = "/projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_large_480_don"
fname = open("./scripts/yaml_dataset.txt", "w")
for fi in os.listdir(path):
    if fi.endswith('.p'):
        name = fi.split('.')[0]
        string = '- "' + name + '"'
        fname.write(string)
        fname.write('\n')


