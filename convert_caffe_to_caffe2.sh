./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt

python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel


(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ ./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt
(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel


(fasterrcnn-py2) majid@majid:~/Myprojects/CAFFE/python$ python vgg16.tf/src/dump_caffemodel_weights.py --caffe_root /home/majid/Myprojects/CAFFE --caffemodel_path /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel --prototxt_path /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy.prototxt --caffe_weights_path VGG_CNN_M_1024.pkl

(fasterrcnn-py2) majid@majid:~/Myprojects/CAFFE/python$ python vgg16.tf/src/dump_caffemodel_weights.py --caffe_root /home/majid/Myprojects/CAFFE --caffemodel_path /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel --prototxt_path /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy.prototxt --caffe_weights_path VGG_ILSVRC_16_layers.pkl
