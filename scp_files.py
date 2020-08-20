import os
root = '/projects/katefgroup/datasets/shamit_carla_correct/npys'
fi = open('mc_4carst.txt','r')
ccnt=0
for line in fi.readlines():
    path = os.path.join(root, line)
    print(path)
    os.system('scp cmu:{} /hdd/shamit/carla_4_cars'.format(path))
    print("done with {} files".format(str(ccnt)))