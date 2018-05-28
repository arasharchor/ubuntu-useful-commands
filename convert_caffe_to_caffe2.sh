./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt

python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel


(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ ./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt
(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel


(fasterrcnn-py2) majid@majid:~/Myprojects/CAFFE/python$ python vgg16.tf/src/dump_caffemodel_weights.py --caffe_root /home/majid/Myprojects/CAFFE --caffemodel_path /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel --prototxt_path /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy.prototxt --caffe_weights_path VGG_CNN_M_1024.pkl

(fasterrcnn-py2) majid@majid:~/Myprojects/CAFFE/python$ python vgg16.tf/src/dump_caffemodel_weights.py --caffe_root /home/majid/Myprojects/CAFFE --caffemodel_path /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel --prototxt_path /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy.prototxt --caffe_weights_path VGG_ILSVRC_16_layers.pkl


(caffe_2) majid@majid:~/Myprojects/Detectron$ python tools/pickle_caffe_blobs.py --prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt --caffemodel /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel --output hey

[u'conv1_b', u'conv1_w', u'conv2_b', u'conv2_w', u'conv3_b', u'conv3_w', u'conv4_b', u'conv4_w', u'conv5_b', u'conv5_w', u'fc6_b', u'fc6_w', u'fc7_b', u'fc7_w', u'fc8_b', u'fc8_w']

(caffe_2) majid@majid:~/Myprojects/Detectron$ python tools/pickle_caffe_blobs.py --prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt --caffemodel /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel --output VGG_ILSVRC_16_layers.pkl
[u'conv1_1_b', u'conv1_1_w', u'conv1_2_b', u'conv1_2_w', u'conv2_1_b', u'conv2_1_w', u'conv2_2_b', u'conv2_2_w', u'conv3_1_b', u'conv3_1_w', u'conv3_2_b', u'conv3_2_w', u'conv3_3_b', u'conv3_3_w', u'conv4_1_b', u'conv4_1_w', u'conv4_2_b', u'conv4_2_w', u'conv4_3_b', u'conv4_3_w', u'conv5_1_b', u'conv5_1_w', u'conv5_2_b', u'conv5_2_w', u'conv5_3_b', u'conv5_3_w', u'fc6_b', u'fc6_w', u'fc7_b', u'fc7_w', u'fc8_b', u'fc8_w']
