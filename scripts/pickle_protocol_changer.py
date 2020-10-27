import os 
import pickle
import ipdb 
st = ipdb.set_trace

root = "/projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_large_480_don"
newroot = "/projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_large_480_don_lowerprot"

total = len(os.listdir(root))
for cnt, pi in enumerate(os.listdir(root)):
    print(f"Processing file: {pi}.... Num {cnt}/{total}")
    if pi.endswith('.p'):
        # st()
        path = os.path.join(root, pi)
        a = pickle.load(open(path, "rb"))
        dumppath = os.path.join(newroot, pi)
        pickle.dump(a, open(dumppath, 'wb'), protocol=2)



