import os
from datetime import datetime
import numpy as np
import argparse
import json
import time

from bashplotlib.histogram import plot_hist

from PIL import Image, ImageOps, ImageDraw, ImageFont
# import cv2  # use either PIL or opencv ..

import tvm
from tvm.contrib import graph_runtime


def cv_save_detections(path, img, pred, colors):
    """
    Draws boxes and saves image in .jpg format.

    :param path: image will be saved here
    :param img: original image
    :param pred: predictions (boxes) in [x1, y1, x2, y2] format
    :param colors: list of colors for each class

    """
    img_h, img_w = img.shape[:2]

    for x1, y1, x2, y2, obj, cls in pred:
        cls_idx = int(np.argmax(cls))
        color = colors[cls_idx]

        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=color, thickness=int(0.002 * img_h))
        cv2.putText(img, "{:.2f} {:.2f}".format(obj, cls), (x1, y1),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=int(0.0006 * img_h), color=(0, 255, 0),
                    thickness=3, lineType=cv2.LINE_AA)

    # resize image down to width of 1024
    base_width = 1024
    wpercent = base_width / float(img_w)
    hsize = int(float(img_h) * float(wpercent))
    img = cv2.resize(img, dsize=(base_width, hsize), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.normpath(path), img)


def pil_save_detections(path, img, pred, colors):
    """
    Draws boxes and saves image in .jpg format.

    :param path: image will be saved here
    :param img: original image
    :param pred: predictions (boxes) in [x1, y1, x2, y2] format
    :param colors: list of colors for each class

    """
    img_w, img_h = img.size

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("old_stuff/arial.ttf", int(0.025 * img_h))
    for x1, y1, x2, y2, obj, cls in pred:
        cls_idx = int(np.argmax(cls))

        # draw boxes and text
        draw.rectangle(((x1, y1), (x2, y2)),
                       outline=colors[cls_idx],
                       width=int(0.002 * img_h))
        draw.text((x1, y1), "{:.2f} {:.2f}".format(obj, cls),
                  fill=(0, 255, 0, 128), font=font)

    # resize image down to width of 1024
    base_width = 1024
    wpercent = base_width / float(img_w)
    hsize = int(float(img_h) * float(wpercent))
    img = img.resize((base_width, hsize), Image.LANCZOS)
    img.save(os.path.normpath(path), quality=95)


def cv_get_image(path, target_size=(416, 416), color=(127, 127, 127)):
    """
    Loads an image and returns a np array of a square, padded image,
    and its original dimensions, using OpenCV.

    :param path: path to image
    :param target_size: new image size
    :param color: the color used for padding
    :param mode: will pad with minimum rectangle multiple of 32 or square

    :return: np.array of RGB channel image with shape (3, width, height)
    """
    img = cv2.imread(path)  # BGR
    original_size = img.shape[:2]  # current shape [height, width]

    # Compute ratio and resize
    ratio = float(max(target_size)) / max(original_size)
    new_size = (int(round(original_size[1] * ratio)), int(round(original_size[0] * ratio)))
    data = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)  # cv2.INTER_LANCZOS4

    # Add padding
    if target_size[0] == target_size[1]:  # square
        delta_w = (target_size[0] - new_size[0]) / 2
        delta_h = (target_size[1] - new_size[1]) / 2
    else:  # minimum rectangle
        delta_w = ((target_size[0] - new_size[0]) % 32) / 2  # width padding
        delta_h = ((target_size[1] - new_size[1]) % 32) / 2  # height padding

    top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
    left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))
    data = cv2.copyMakeBorder(data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square

    # window_name = "img pad resize {}".format(data.shape)
    # cv2.startWindowThread()
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # cv2.imshow(window_name, img)
    # cv2.waitKey()

    data = data[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    data = np.ascontiguousarray(data, dtype=np.float32) / 255.0  # uint8 to float32

    assert data.shape[1] == target_size[1], \
        "Image aspect ratio {} incompatible with network receptive field {}!"\
            .format(original_size, target_size)

    return img, data, (delta_w, delta_h)


def pil_get_image(path, target_size=(416, 416), color=(127, 127, 127)):
    """
    Loads an image and returns a np array of a square, padded image,
    and its original dimensions, using Pillow.

    :param path: path to image
    :param target_size: new image size
    :param color: the color used for padding
    :return: np.array of RGB channel image with shape (channels, height, width)
    """
    img = Image.open(path)
    original_size = img.size

    # Compute ratio and resize
    ratio = float(max(target_size)) / max(original_size)
    new_size = tuple([int(round(x * ratio)) for x in original_size])
    data = img.resize(new_size, resample=Image.LANCZOS)

    # Add padding
    if target_size[0] == target_size[1]:  # square
        delta_w = target_size[0] - new_size[0]
        delta_h = target_size[1] - new_size[1]
    else: # minimum rectangle
        delta_w = (target_size[0] - new_size[0]) % 32
        delta_h = (target_size[1] - new_size[1]) % 32

    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    data = ImageOps.expand(data, padding, fill=color)

    # (H, W, C) view as (C, H, W) and set values from 0 - 255 to 0.0 - 1.0
    data = np.array(data).astype(np.float32).transpose((2, 0, 1)) / 255.0

    assert data.shape[1] == target_size[1], \
        "Image aspect ratio {} incompatible with network receptive field {}!"\
            .format(original_size, target_size)

    return img, data, (delta_w, delta_h)


def bbox_iou(box1, box2):
    """
    Computes IoU between bounding boxes in format x1 y1, x2, y2

    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    : ndarray
        (N, M) shaped array with IoUs
    """

    if len(box1.shape) < 2:
        box1 = np.expand_dims(box1, axis=0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - \
         np.maximum(np.expand_dims(box1[:, 0], 1), box2[:, 0])
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - \
         np.maximum(np.expand_dims(box1[:, 1], 1), box2[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims(area1, axis=1) + area2 - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    return np.squeeze( (iw * ih) / ua )


def xywh2xyxy(x):
    """
    Convert bounding box format
    :param x: [x, y, w, h]
    :return: [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh(x):
    """
    Convert bounding box format

    :param x: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def rescale_box_coords(pred, full_size, rf_size, rf_pad):
    """
    Rescales box coordinates to original image size.

    :param pred: predictions in absolute coordinates (x, y, w, h)
    :param full_size: original, full image size (w, h)
    :param rf_size: size of receptive field (w, h)
    :param rf_pad: padding used to fit image for rf size

    :return: rescaled coordinates in (x1, y1, x2, y2)
    """
    # Change coordinate system
    pred[:, :4] = xywh2xyxy(pred[:, :4])

    # Rescale to receptive field + rf padding
    pred[:, [0, 2]] *= rf_size[0]
    pred[:, [1, 3]] *= rf_size[1]

    # Remove receptive field padding
    pred[:, [0, 2]] -= (rf_pad[0] / 2)
    pred[:, [1, 3]] -= (rf_pad[1] / 2)

    # Rescale to original image size (full)
    ratio = max(full_size) / max(rf_size)
    pred[:, :4] *= ratio

    return pred


def non_max_suppression(pred, input_shape, original_shape,
                        conf_thres=0.1, nms_thres=0.5, min_wh=5):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.

    Discard all boxes with object conf threshold <= conf_thres
    While there are any remaining boxes :
        1) Pick the box with the largest object conf thres
        2) Output that as a prediction
        3) Discard any remaining boxes with IOU>=0.5 with the box output in the previous step

    :param pred: predictions in format (x, y, w, h, obj_conf, c1_conf, ..., cN_conf)
    :param conf_thres: confidence score threshold
    :param nms_thres: non max suppression threshold
    :param min_wh: (pixels) minimum box width and height

    :return: Detections with shape: (x1, y1, x2, y2, obj_conf, c1_conf, ..., cN_conf)
    """

    # Rescale to input image size (coords in x y w h)
    pred[:, [0, 2]] *= input_shape[0]
    pred[:, [1, 3]] *= input_shape[1]

    # Select only suitable predictions (also filters w, h)
    keep = (pred[:, 4] > conf_thres) & \
           (pred[:, 2:4] > min_wh).all(1) & \
           np.isfinite(pred).all(1)
    pred = pred[keep]

    # if there are no boxes left, return an empty list
    if pred.size == 0:
        return np.array([])

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    pred[:, :4] = xywh2xyxy(pred[:, :4])

    # Rescale coordinates to full (unpadded) original image
    # Now NMS running in original image coordinates ...
    pred = rescale_coords(pred, input_shape, original_shape)

    # Get detections sorted by decreasing confidence scores
    pred = pred[(-pred[:, 4]).argsort()]

    # Perform non-maximum suppression
    max_detections = []
    while pred.shape[0]:
        # Get detection with highest confidence and save as max detection
        max_detections.append(pred[0])
        # Stop if we're at the last detection
        if len(pred) == 1:
            break
        # Get the IOUs for all boxes with lower confidence
        ious = bbox_iou(max_detections[-1], pred[1:])
        # Remove detections with IoU >= NMS threshold
        pred = pred[1:][ious < nms_thres]

    return np.vstack(max_detections)


def non_max_suppression_fast(pred, original_shape, input_shape, pad,
                             conf_thres=0.1, nms_thres=0.5, min_wh=0.0008):
    """
    Performs non-maximum suppression.

    wider-face (overall)        MIN W 0.0009765625 MIN H 0.0007685551
    classroom test_images (4:3)  MIN W 0.004289     MIN H 0.009804

    :param pred: list of predictions in following format:
            (cls_idx, x, y, w, h, obj_conf, cls1_conf, cls2_conf, ..., clsN_conf)
            in absolute coordinates format.
    :param input_shape: shape of network receptive field (w, h)
    :param original_shape:  shape of original input image (w, h)
    :param pad: padding used to fit to receptive field (w, h)
    :param conf_thres: confidence threshold to filter out low confidence boxes
    :param nms_thres: will filter out overlapping duplicates
    :param min_wh: boxes with width or height lower than this will be removed

    :return: predictions with boxes in (x1, y1, x2, y2) format, rest is the same
    """

    # Select only suitable predictions (also filters w, h)
    keep = (pred[:, 4] > conf_thres) & \
           (pred[:, 2:4] > min_wh).all(1) & \
           np.isfinite(pred).all(1)
    pred = pred[keep]

    # if there are no boxes left, return an empty list
    if pred.size == 0:
        return np.array([])

    # Rescale boxes to original image size
    pred = rescale_box_coords(pred, original_shape, input_shape, pad)

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if pred.dtype.kind == "i":
        pred = pred.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # if probabilities are provided, sort on them instead

    # sort the indexes
    idxs = np.argsort(pred[:, 4])

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_thres)[0])))

    # return only the bounding boxes that were picked
    return pred[pick]  #.astype("int")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

            # Plot
            # plt.plot(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_stats(pred, labels, stats, opt):
    """ Statistics per image """

    tcls = labels[:, 0].astype(np.int32).tolist()

    # Assign all predictions as incorrect
    correct = [0] * len(pred)

    # Get class idx from pdist
    pred_cls = np.argmax(pred[:, 5:], axis=1)

    detected = []
    tbox = xywh2xyxy(labels[:, 1:5])

    # Search for correct predictions
    for i, (*pbox, objp, clsp) in enumerate(pred):
        # Break if all targets already located in image
        if len(detected) == len(labels):
            break

        # Get idx of class pdist
        pcls = pred_cls[i]

        # Continue if predicted class not among image classes
        if pcls not in tcls:
            continue

        # Compare only with same class as predicted
        same = [pcls == t for t in tcls]

        # Best iou, index between pred and targets
        ious = bbox_iou(np.array(pbox), tbox[same])
        iou = np.max(ious)
        bi = int(np.argmax(ious))

        # If iou > threshold and class is correct mark as correct
        if iou > opt.iou_thres and bi not in detected:
            correct[i] = 1
            detected.append(bi)

    # Append statistics (correct, conf, pcls, tcls)
    stats.append((correct, pred[..., 4], pred_cls, tcls))

    return stats


def detect(opt):
    """ Runs detections """

    # list of class labels
    class_names = ['face', 'other']
    colors = ['red', 'yellow']

    fullpath = os.path.join(opt.dir, opt.model)
    print("Loading model files from: {}.*".format(fullpath))

    # --- start loading net ---

    # load tvm model files
    if os.path.isfile(fullpath + ".so"):
        # Model library is compiled as dynamic library
        lib = tvm.module.load(fullpath + ".so")
    elif os.path.isfile(fullpath + ".tar"):
        # Model will be compiled locally (.tar.so) when module.load is called
        lib = tvm.module.load(fullpath + ".tar")
    elif os.path.isfile(fullpath + ".o"):
        # Model library is compiled as system library mode
        lib = tvm.module.load(fullpath + ".o")
    else:
        raise Exception("Couldn't find model lib file at {}".format(fullpath))

    graph = open(fullpath + ".json").read()
    params = bytearray(open(fullpath + ".params", "rb").read())

    # load target str and set runtime context
    with open(fullpath + ".target") as f:
        target_str = f.readline()
        print("Compiled model target is:", target_str)
    target = tvm.target.create(target_str)
    ctx = tvm.context(str(target), 0)

    # create module runtime
    net = graph_runtime.create(graph, lib, ctx)
    net.load_params(params)

    # Load input name, shape dtype from json file
    g = json.loads(graph)
input_name = g['nodes'][0]['name']
batch_size, channels, height, width = g['attrs']['shape'][1][0]
dtype = g['attrs']['dltype'][1][0]

print("Input name '{}' shape: (N {}, C {}, W {}, H {}) dtype: {} (from json)"
      .format(input_name, batch_size, channels, width, height, dtype))

# --- end loading net ---

# setup output folders
config = "{}_conf{}_nms{}".format(opt.model, opt.conf_thres, opt.nms_thres)
    savedir = os.path.normpath("{}/{}".format(opt.output, config))
    os.makedirs(savedir, exist_ok=True)
    
    if not opt.evaluate:
        obj_confs_pre, obj_confs_post, class_confs_post = [], [], []
else:
    seen, stats = 0, []
    
    # traverse folder and load images and labels
    for root, dirs, files in os.walk(opt.images, topdown=True):
        # do not recurse if output files written in sub-folders
        if os.path.normpath(opt.output) in root:
            continue
        for name in sorted(files):
            if not name.endswith((".jpg", ".png", ".jpeg", ".bmp")):
                continue
            img_path = os.path.join(root, name)
            
            imgstart = datetime.now()
            img, data, rf_pad = pil_get_image(img_path, (width, height))
            img_w, img_h = img.size
            _, rf_h, rf_w = data.shape
            print("Loading image {} \ntook: {}".format(img_path, datetime.now() - imgstart))
            
            runstart = datetime.now()
            print("Running inference ... ", end='')
            net.set_input(input_name, tvm.nd.array(data.astype(dtype)))
            net.run()
            pred = net.get_output(0).asnumpy()
            print("took {}".format(datetime.now() - runstart))
            
            if not opt.evaluate:
                # keep a list of object confidences before NMS
                obj_confs_pre.extend(np.squeeze(pred[..., 4]).tolist())

            # Rescale outputs to original image size and do NMS
            nmsstart = datetime.now()
            print("Running NMS ... ", end='')
            pred = np.squeeze(pred)
            pred = non_max_suppression_fast(pred, (img_w, img_h), (rf_w, rf_h), rf_pad,
                                            opt.conf_thres, opt.nms_thres)
            print("took {}".format(datetime.now() - nmsstart))

            # continue to next item if no detections left
            num_dets = np.shape(pred)[0]
            print("Num. detections:", num_dets)
            if num_dets == 0:
                continue

            if opt.evaluate:
                label_path = ".".join(img_path.split('.')[:-1]) + ".txt"
                if not os.path.isfile(label_path):
                    continue

                with open(label_path, 'r') as f:
                    targets = np.array([x.split() for x in f.read().splitlines()],
                                       dtype=np.float32)
                if targets.size == 0:
                    continue
                else:
                    # Rescale to full image coordinates
                    targets[:, [1, 3]] *= img_w
                    targets[:, [2, 4]] *= img_h

                # keep running stats for all images
                stats = compute_stats(pred, targets, stats, opt)
                seen += 1
            else:
                # keep a list of object confidences and class confidences after NMS
                obj_confs_post.extend(pred[..., 4].tolist())
                class_confs_post.append(pred[..., 5:].tolist())

                # save images if requested as .jpg
                savepath = "{}/{}".format(savedir, os.path.basename(img_path))
                savepath = os.path.normpath(".".join(savepath.split(".")[:-1]) + ".jpg")

                # draw boxes, saving image
                if not opt.no_plot:
                    drawstart = datetime.now()
                    pil_save_detections(savepath, img, pred, colors)
                    print("Drawing boxes and saving image took {}".format(datetime.now() - drawstart))

                # save detections to .txt file with same name
                writestart = datetime.now()
                savetxt = savepath.replace(".jpg", ".txt")
                with open(savetxt, "a") as f:
                    for *box, _, cls in pred:
                        cls_idx = int(np.argmax(cls))
                        x, y, w, h = xyxy2xywh(np.expand_dims(box, 0)).squeeze()
                        x = min(1, max(0, x / img_w))
                        y = min(1, max(0, y / img_h))
                        w = min(1, max(0, w / img_w))
                        h = min(1, max(0, h / img_h))
                        f.write("{:d} {:.10f} {:.10f} {:.10f} {:.10f}\n"
                                .format(cls_idx, x, y, w, h))
                print("Writing detections to file took {}".format(datetime.now() - writestart))

            print("Processing image, TOTAL took {}".format(datetime.now() - imgstart))
            print("*" * 80)

    if not opt.evaluate:
        # Show some stats
        plot_hist(obj_confs_pre, height=5, bincount=15,
                  showSummary=True, xlab=True, title="Objectness Confidence Pre-NMS")
        plot_hist(obj_confs_post, height=5, bincount=15,
                  showSummary=True, xlab=True, title="Objectness Confidence Post-NMS")
        class_confs_post = np.vstack(class_confs_post)
        for c in range(np.shape(class_confs_post)[1]):
            plot_hist(class_confs_post[:, c], height=5, bincount=15,
                      showSummary=True, xlab=True,
                      title="Class '{}' Confidence Post-NMS".format(class_names[c]))

        return None, None, None, None, {}
    else:
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
        nc = len(class_names)  # num classes
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
        if len(stats):
            p, r, ap, f1, ap_class = ap_per_class(*stats)
            m_p, m_r, m_ap, m_f1 = p.mean(), r.mean(), ap.mean(), f1.mean()

        # Print to file as well
        f = open(savedir + "/stats_log.txt", "w")

        # Print results
        print(('%10s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'))
        print(('%10s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1'), file=f)
        pf = '%10s' + '%10.3g' * 6  # print format
        print(pf % ('all', seen, nt.sum(), m_p, m_r, m_ap, m_f1), end='\n\n')
        print(pf % ('all', seen, nt.sum(), m_p, m_r, m_ap, m_f1), end='\n\n', file=f)

        # Print results per class
        if nc > 1 and len(stats):
            per_class = {'AP': {}, 'P': {}, 'R': {}, 'F1': {}}
            for i, c in enumerate(ap_class):
                per_class['AP'][class_names[c]] = ap[i]
                per_class['P'][class_names[c]] = p[i]
                per_class['R'][class_names[c]] = r[i]
                per_class['F1'][class_names[c]] = f1[i]
                print(pf % (class_names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))
                print(pf % (class_names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]), file=f)
        else:
            per_class = {}
        f.close()

        return m_p, m_r, m_ap, m_f1, per_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./compiled/",
                        help="name of folder containing saved files")
    parser.add_argument("--model", type=str, default="y3t-c1-rf416-b1.onnx.rk3399.arm_cpu.float32",
                        help="name of model")
    parser.add_argument('--iou-thres', type=float, default=0.5,
                        help='minimum box overlap percentage to qualify as detected (Default: 0.5)')
    parser.add_argument("--conf-thres", type=float, default=0.2,
                        help="confidence threshold to filter false positives (Default: 0.2)")
    parser.add_argument("--nms-thres", type=float, default=0.5,
                        help="IOU threshold for non-maximum suppression (Default: 0.5)")
    parser.add_argument('--images', type=str,
                        default='./samples/detector/',
                        help='path to images')
    parser.add_argument('--output', type=str,
                        default='./output/detector/',
                        help='path where images with detections are saved')
    parser.add_argument('--evaluate', action='store_true',
                        help='get mAP based on labels located at same location with images.')
    parser.add_argument('--no-plot', action='store_true',
                        help='only detections .txt files will be written to output folder.')
    opt, _ = parser.parse_known_args()
    print(opt)
    tic = datetime.now()
    detect(opt)
    print("All took", datetime.now() - tic)
    mean_inference_speed = datetime.now() - tic/ seen
    print("Mean_inference" ,mean_inference_speed)

