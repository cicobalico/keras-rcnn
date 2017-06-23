import keras
import keras_rcnn.backend
import numpy
import tensorflow

def separate_pred(y_pred):
    n_anchors = tensorflow.shape(y_pred)[-1] // 5
    return y_pred[:, :, :, 0 : 4 * n_anchors], y_pred[:, :, :, 4 * n_anchors : 8 * n_anchors]


def encode(features, image_shape, y_true, stride = 16):
    # get all anchors inside bbox
    shifted_anchors = keras_rcnn.backend.shift(features, stride)
    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(shifted_anchors, image_shape)

    # indices of gt boxes with the greatest overlap, bbox labels
    # TODO: assert y_true.shape[0] == 1
    y_true = tensorflow.gather(y_true, tensorflow.constant(0))
    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(y_true, all_inside_bbox, inds_inside)

    # gt boxes
    gt_boxes = tensorflow.gather(y_true, argmax_overlaps_inds)

    # Convert fixed anchors in (x, y, w, h) to (dx, dy, dw, dh)
    bbox_reg_targets = keras_rcnn.backend.bbox_transform(all_inside_bbox, gt_boxes)

    return bbox_labels, bbox_reg_targets, inds_inside, tensorflow.shape(shifted_anchors)[0]


def proposal(anchors, *args, **kwargs):
    def f(y_true, y_pred):
        # separate y_pred into rpn_cls_pred and rpn_reg_pred
        y_pred_regression, y_pred_classification = separate_pred(y_pred)

        # convert y_true from gt_boxes to gt_anchors
        if 'image_shape' in kwargs:
            image_shape = kwargs['image_shape']
        if 'stride' in kwargs:
            stride = kwargs['stride']

        features = tensorflow.shape(y_pred)[1:3]

        gt_classification, gt_regression, inds_inside, num_shifted_anchors = encode(features, image_shape, y_true, stride)

        N = features[0] * features[1] * anchors
        ii = tensorflow.constant(0)
        initial_y_true_classification = tensorflow.TensorArray(size = N, dtype = tensorflow.float32)
        initial_y_true_regression1 = tensorflow.TensorArray(size = N * 4, dtype = tensorflow.float32)
        initial_y_true_regression2 = tensorflow.TensorArray(size = N * 4, dtype = tensorflow.float32)

        def cond(ii, *args):
            return ii < N

        def body(ii, c, r1, r2):
            def not_inds(ii, c, r1, r2):
                c = c.write(ii, tensorflow.cast(0, tensorflow.float32))
                r1 = r1.write(ii * 4, tensorflow.cast(0, tensorflow.float32))
                r1 = r1.write(ii * 4+1, tensorflow.cast(0, tensorflow.float32))
                r1 = r1.write(ii * 4+2, tensorflow.cast(0, tensorflow.float32))
                r1 = r1.write(ii * 4+3, tensorflow.cast(0, tensorflow.float32))
                r2 = r2.write(ii * 4, tensorflow.cast(0, tensorflow.float32))
                r2 = r2.write(ii * 4+1, tensorflow.cast(0, tensorflow.float32))
                r2 = r2.write(ii * 4+2, tensorflow.cast(0, tensorflow.float32))
                r2 = r2.write(ii * 4+3, tensorflow.cast(0, tensorflow.float32))
                return ii+1, c, r1, r2
        
            def in_inds(ii, c, r1, r2):
                idx = tensorflow.cast(tensorflow.where(tensorflow.equal(inds_inside, ii))[0][0], tensorflow.int32)
                gt_class = gt_classification[idx]

                c = c.write(ii, tensorflow.cast(gt_class, tensorflow.float32))
                r1 = r1.write(ii * 4, tensorflow.cast(gt_class, tensorflow.float32))
                r1 = r1.write(ii * 4+1, tensorflow.cast(gt_class, tensorflow.float32))
                r1 = r1.write(ii * 4+2, tensorflow.cast(gt_class, tensorflow.float32))
                r1 = r1.write(ii * 4+3, tensorflow.cast(gt_class, tensorflow.float32))
                r2 = r2.write(ii * 4, gt_regression[idx, 0])
                r2 = r2.write(ii * 4+1, gt_regression[idx, 1])
                r2 = r2.write(ii * 4+2, gt_regression[idx, 2])
                r2 = r2.write(ii * 4+3, gt_regression[idx, 3])
                return ii+1, c, r1, r2
        
            return tensorflow.cond(tensorflow.less(tensorflow.shape(tensorflow.where(tensorflow.equal(inds_inside, ii)))[0], 1), lambda: not_inds(ii, c, r1, r2), lambda: in_inds(ii, c, r1, r2))

        index, y_true_classification, r1, r2 = tensorflow.while_loop(
                cond, 
                body, 
                [ii, initial_y_true_classification, initial_y_true_regression1, initial_y_true_regression2]
        )

        y_true_classification = y_true_classification.stack()
        r1 = r1.stack()
        r2 = r2.stack()
        y_true_regression = tensorflow.concat([tensorflow.reshape(r1, (1, features[0], features[1], anchors*4)), tensorflow.reshape(r2, (1, features[0], features[1], anchors*4))], axis=3)
        y_true_classification = tensorflow.reshape(y_true_classification, (1, features[0], features[1], anchors))
        y_true_classification = tensorflow.concat([y_true_classification, y_true_classification], axis=3)

      
        classification = _classification(anchors=anchors)(y_true_classification, y_pred_classification)

        regression = _regression(anchors=anchors)(y_true_regression, y_pred_regression)

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
