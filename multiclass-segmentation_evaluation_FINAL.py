import sys
import os
import re
import numpy as np
import eval_segm
import cv2

# CHANGELOG
# 04.06.2018 Corentin
# - changed results print from raw string to formatted strings
# - changed some lines to comply with PEP8 style guidelines (imports, spaces, variable names, etc.)
# - reordered some lines like constants declarations
# - separated global and per-image results print
DLR = True
if DLR:
    home = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/'
else:
    home = '/home/majid/Myprojects/Semantic-Segmentation-Suite/Test/'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-False-None-DeepLabV3_plus-Res101'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-DeepLabV3_plus-Res101'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-False-None-DeepLabV3-Res50'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-False-None-FC-DenseNet103'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-False-None-PSPNet-Res101'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-DeepLabV3-Res50'
# GT_DIR = home + 'SingleVehicleDLR_1024-True-None-Encoder-Decoder'
# GT_DIR = home + 'AerialLane18_512-False-None-FRRN-A'
# GT_DIR = home + 'AerialLane18_1024-False-None-DeepLabV3_plus-Res50'
# GT_DIR = home + 'AerialLane18_1024-False-None-DeepLabV3_plus-Res101'
# GT_DIR = home + 'AerialLane18_1024_aug-False-None-DeepLabV3_plus-Res101'
# GT_DIR = home + 'AerialLane18_1024_aug-False-None-GCN-Res101'
GT_DIR = home + 'AerialLane18_1024_aug-False-None-DeepLabV3-Res101'
/media/azim_se/MeinDaten/DATASETS/DLR/Delivered-MultiClassLane-SkyScapes/test/labels

# GT_DIR = home + 'AerialLane18_bestresults'
# GT_DIR = home + 'AerialLane18_test_gt'
# GT_DIR = home + 'AerialLane18_1024-False-None-DeepLabV3_plus-Res152'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-Encoder-Decoder-Skip'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-FC-DenseNet103'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-GCN-Res101'
# GT_DIR = '/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-RefineNet-Res101'

# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_fcn_8s_vgg19_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_pretrained_dense_aspp_densenet_121_bc_LR_3e-4_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_pretrained_fcn_8s_resnet50_LR_1e-1_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_pretrained_fcn_8s_resnet50_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_pretrained_fcn_8s_resnet101_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/1_class_pretrained_fcn_8s_vgg19_LR_1e-3_LRD_1.00_XENTx1/pred/test'
PRED_DIR = GT_DIR #'/home/azim_se/MyProjects/Semantic-Segmentation-Suite/Test/SingleVehicleDLR_1024-True-None-DeepLabV3_plus-Res101'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_fcn_8s_vgg19_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_dense_aspp_densenet_121_bc_LR_3e-4_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_fcn_8s_resnet50_LR_1e-2_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_fcn_8s_resnet50_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_fcn_8s_resnet101_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_fcn_8s_vgg19_LR_1e-3_LRD_1.00_XENTx1/pred/test'
# PRED_DIR = '/home/henr_co/workspace/MajidVehicleSegmentation/2_classes_pretrained_pspnet_resnet50_LR_1e-3_XENTx1/pred/test'
OPEN = False  # to remove small boxPoints
DILATE = False # to make it bigger
MINREC = False
DRAW = False
MIN_AREA = 100 #100.0
NUM_CLASS = 1

gt_count = {
    '2012-04-26-Muenchen-Tunnel_4K0G0110': 954,
    '2012-04-26-Muenchen-Tunnel_4K0G0120': 1175,
    '2012-04-26-Muenchen-Tunnel_4K0G0130': 646,
    '2012-04-26-Muenchen-Tunnel_4K0G0140': 507,
    '2012-04-26-Muenchen-Tunnel_4K0G0150': 898,
    '2012-04-26-Muenchen-Tunnel_4K0G0160': 609,
    '2012-04-26-Muenchen-Tunnel_4K0G0250': 451,
    '2012-04-26-Muenchen-Tunnel_4K0G0265': 177,
    '2012-04-26-Muenchen-Tunnel_4K0G0278': 169,
    '2012-04-26-Muenchen-Tunnel_4K0G0285': 342
}
gt_counts = [954, 1175, 646, 507, 898, 609, 451, 177, 169, 342]
gts_path = np.sort([name for name in os.listdir(GT_DIR) if name.endswith('_gt.png')])
preds_path = np.sort([name for name in os.listdir(PRED_DIR) if name.endswith('_pred.png')])
print(gts_path)
print(preds_path)

total_gt_counts = 5928
pa_list = []
ma_list = []
list_rate_precision_mean = []
iu_list = []
m_iu_list = []
fw_iu_list = []
num_true_positives = 0
num_false_positives = 0
counts = []
acc = np.zeros(10)
kernel = np.ones((3, 3), np.uint8)

for i, gt_path in enumerate(gts_path):
    print('###################')
    count = 0
    _name = os.path.splitext(gt_path)[0][:-10]
    # _name = os.path.splitext(gt_path)[0][:-3]
    print("Evaluating image: " + str(_name))
    label = cv2.imread(GT_DIR + '/' + _name + '_0_main_gt.png')
    pred = cv2.imread(PRED_DIR + '/' + _name + '_0_main_pred.png')
    # label = cv2.imread(GT_DIR + '/' + _name + '_gt.png')
    # pred = cv2.imread(PRED_DIR + '/' + _name + '_gt.png')
    if len(np.shape(pred)) > 1:
        print('converting to grayscale')
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        print('unique label before is {}'.format(np.unique(label)))
        print('unique pred before is {}'.format(np.unique(pred)))
        cls_1 = pred == 79
        label_cls_1 = label == 79
        # cls_1 = pred == 255
        # label_cls_1 = label == 255
        if NUM_CLASS == 1:
            pred[cls_1] = 1
            label[label_cls_1] = 1
        else:
            pred[cls_1] = 1
            pred[cls_2] = 2
    else:
        pred = pred_

    if MINREC:
        ret, thresh = cv2.threshold(pred, 0, 9, 0)
        _, contours, threshold = cv2.findContours(thresh, 1, 2)
        mask = np.zeros(pred.shape[:2], dtype="uint8")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print('area is:' + str(area))
            if area > MIN_AREA:
                count += 1
                rect = cv2.minAreaRect(cnt)
                # x,y,w,h = cv2.boundingRect(cnt)
                # charImg = mask[y:y+h, x:x+w]
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                # print('box is:{}'.format(box))
                # print('cnt is:{}'.format(cnt))
                if DRAW:
                    cv2.drawContours(mask, [box], 0, int(pred[int(rect[0][1])][int(rect[0][0])]), cv2.FILLED)

    if DRAW:
        cv2.imwrite("%s_threshold.png"%(_name),np.uint8(mask)*255) #cv2.cvtColor(np.uint8(mask), cv2.COLOR_RGB2BGR)cv2.COLOR_BGR2GRAY


    if OPEN is True:
        pred = np.array(cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel))
    if DILATE is True:
        pred = np.array(cv2.dilate(pred, kernel, iterations=1))
    """
    cls_1 = pred == 1
    cls_3 = pred == 3
    cls_2 = pred == 2
    cls_4 = pred == 4
    cls_5 = pred == 5
    cls_6 = pred == 6
    cls_7 = pred == 7
    cls_8 = pred == 8
    pred[cls_3] = 1
    pred[cls_2] = 2
    pred[cls_4] = 2
    pred[cls_5] = 2
    pred[cls_6] = 2
    pred[cls_7] = 2
    pred[cls_8] = 2
    """
    print('unique label after is {}'.format(np.unique(label)))
    print('unique pred after is {}'.format(np.unique(pred)))

    pa = eval_segm.pixel_accuracy(pred, label)
    ma = eval_segm.mean_accuracy(pred, label)
    rate_precision_mean = eval_segm.mean_precision(pred, label)
    m_iu, iu = eval_segm.mean_IU(pred, label)
    fw_iu = eval_segm.frequency_weighted_IU(pred, label)
    pa_list.append(pa)
    ma_list.append(ma)
    list_rate_precision_mean.append(rate_precision_mean)
    iu_list.append(iu)
    m_iu_list.append(m_iu)
    fw_iu_list.append(fw_iu)
    num_true_positives += eval_segm.get_num_true_positives(pred, label)
    num_false_positives += eval_segm.get_num_false_positives(pred, label)

    counts.append(count)
    print(np.array(count), np.array(gt_count[_name]))
    acc[i] = 1 - ((np.abs(np.array(count) - np.array(gt_count[_name]))) / np.array(gt_count[_name]))
    # print("pixel_accuracy: " + str(pa))
    # print("mean_accuracy: " + str(ma))
    # print("IU: " + str(iu))
    # print("frequency_weighted: " + str(fw_iu))
    # print("mean_IU: " + str(m_iu))
    # print("##################################")
print("#########################################################")
print("Results for folder: " + PRED_DIR)
print("GLOBAL RESULTS")
print("pixel_accuracy: {:.2f}%".format(np.mean(pa_list) * 100))
print("mean_accuracy (mean_recall): {:.2f}%".format(np.mean(ma_list) * 100))
print("mean_precision: {:.2f}%".format(np.mean(list_rate_precision_mean) * 100))
print("total_true_positives: {:d}".format(num_true_positives))
print("total_false_positives: {:d}".format(num_false_positives))
print("frequency_weighted: {:.2f}%".format(np.mean(fw_iu_list) * 100))
print("mIU for each class is: [{}]".format(", ".join("{:.2f}%".format(score) for score in (np.mean(iu_list, axis=0) * 100))))
print("mean_IU: {:.2f}%".format(np.mean(m_iu_list) * 100))
print("pred total count accuracy is: {:.2f}%".format(np.mean(acc) * 100))
print("PER-IMAGE RESULTS")
str_list_filename = ""
str_list_column = ""
str_list_result = ""
acc_percent = acc * 100
ma_list_percent = np.asarray(ma_list) * 100
list_percent_precision_mean = np.asarray(list_rate_precision_mean) * 100
for i, filename in enumerate(gt_count.keys()):
    str_list_filename += "{:<36}, ".format(filename)
    str_list_column += "{:<5}, {:<10}, {:<9}, {:<6}, ".format("count", "count acc.", "recall", "precision")
    str_list_result += "{:<5d}, {:<10.2f}, {:<9.2f}, {:<6.2f}, ".format(
        counts[i],
        acc_percent[i],
        ma_list_percent[i],
        list_percent_precision_mean[i])
print(str_list_filename)
print(str_list_column)
print(str_list_result)
# print("pred counts are: {}".format(counts))
# print("pred counts accuracies are: [{}]".format(", ".join("{:.2f}%".format(score) for score in (acc * 100))))
# print("gt counts are: {}".format(gt_counts))
