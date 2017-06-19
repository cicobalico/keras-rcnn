import keras
import keras_rcnn.backend
import numpy

def separate_pred(y_pred):
    n_anchors = y_pred.shape[-1] // 5
    return y_pred[:, :, :, -4*n_anchors:], y_pred[:, :, :, :-4*n_anchors]


def encode(features, image_shape, y_true, stride = 16):
    # get all anchors inside bbox
    shifted_anchors = keras_rcnn.backend.shift(features, stride)
    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(shifted_anchors, image_shape)

    # indices of gt boxes with the greatest overlap, bbox labels
    # TODO: assert y_true.shape[0] == 1
    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(y_true[0], all_inside_bbox, inds_inside)

    # gt boxes
    gt_boxes = y_true[0, argmax_overlaps_inds]

    # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox, gt_boxes)

    return bbox_labels, bbox_reg_targets, inds_inside, len(shifted_anchors)


def proposal(anchors, *args, **kwargs):
    def f(y_true, y_pred):
        # separate y_pred into rpn_cls_pred and rpn_reg_pred
        y_pred_regression, y_pred_classification = separate_pred(y_pred)

        # convert y_true from gt_boxes to gt_anchors
        if 'image_shape' in kwargs:
            image_shape = kwargs['image_shape']
        if 'stride' in kwargs:
            stride = kwargs['stride']

        features = y_pred.shape[1:3]

        gt_classification, gt_regression, inds_inside, num_shifted_anchors = encode(features, image_shape, y_true, stride)

        y_true_classification = numpy.zeros((1, features[0], features[1], anchors*2))
        y_true_regression = numpy.zeros((1, features[0], features[1], anchors*4*2))

        for ii in numpy.arange(len(inds_inside)):
            i = inds_inside[ii] // (9*14)
            j = (inds_inside[ii] // 9) % 14
            a = inds_inside[ii] % 9
            y_true_classification[:, i, j, a * gt_classification[ii]] = gt_classification[ii]
            y_true_classification[:, i, j, anchors + a * gt_classification[ii]] = gt_classification[ii]
            y_true_regression[:, i, j, a * gt_classification[ii] * 4 : (a * gt_classification[ii] + 1) * 4] = gt_classification[ii]
            y_true_regression[:, i, j, anchors * 4 + a * gt_classification[ii] * 4 : anchors * 4 + (a * gt_classification[ii] + 1) * 4] = gt_regression[ii, :]

        classification = _classification(anchors=anchors)(keras.backend.variable(y_true_classification), keras.backend.variable(y_pred_classification))

        regression = _regression(anchors=anchors)(keras.backend.variable(y_true_regression), keras.backend.variable(y_pred_regression))

        return classification + regression

    return f


def _classification(anchors=9):
    """
    Return the classification loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function for region propose classification.
    """

    def f(y_true, y_pred):
        # Binary classification loss
        x, y = y_pred[:, :, :, :], y_true[:, :, :, anchors:]

        a = y_true[:, :, :, :anchors] * keras.backend.binary_crossentropy(x, y)
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + y_true[:, :, :, :anchors]
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f


def _regression(anchors=9):
    """
    Return the regression loss of region proposal network.

    :param anchors: Integer, number of anchors at each sliding position. Equal to number of scales * number of aspect ratios.

    :return: A loss function region propose regression.
    """

    def f(y_true, y_pred):
        # Robust L1 Loss
        x = y_true[:, :, :, 4 * anchors:] - y_pred

        mask = keras.backend.less_equal(keras.backend.abs(x), 1.0)
        mask = keras.backend.cast(mask, keras.backend.floatx())

        a_x = y_true[:, :, :, :4 * anchors]

        a_y = mask * (0.5 * x * x) + (1 - mask) * (keras.backend.abs(x) - 0.5)

        a = a_x * a_y
        a = keras.backend.sum(a)

        # Divided by anchor overlaps
        b = keras.backend.epsilon() + a_x
        b = keras.backend.sum(b)

        return 1.0 * (a / b)

    return f
