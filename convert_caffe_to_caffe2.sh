./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt

python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_CNN_M_1024/VGG_CNN_M_1024.caffemodel


(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ ./upgrade_net_proto_text /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt
(caffe_2) majid@majid:~/Myprojects/pytorch/compiled/lib/python2.7/site-packages/caffe2/python$ python -m caffe2.python.caffe_translator /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/deploy_v1.prototxt /home/majid/Myprojects/CAFFE/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel


# use convert tool from caffe-tensorflow to convert caffe model to tensorflow format
python convert.py /data/ken/models/VGG_ILSVRC_16_layers_deploy.prototxt /data/ken/models/VGG_ILSVRC_16_layers.caffemodel /data/ken/models/vgg16_conv.pkl /data/ken/models/vgg16_conv.py

# use dump_network_hdf5 tool from mocha to convert caffe model to hdf5
./dump_network_hdf5 /data/ken/models/VGG_ILSVRC_16_layers_deploy.prototxt /data/ken/models/VGG_ILSVRC_16_layers.caffemodel /data/ken/models/vgg16_conv.hdf5
