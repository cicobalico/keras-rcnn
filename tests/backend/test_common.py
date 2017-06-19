import keras_rcnn.backend
import numpy
import numpy.testing
import keras
import tensorflow

def test_anchor():
    x = tensorflow.convert_to_tensor(numpy.array(
      [[ -84.,  -40.,  99.,  55.],
       [-176.,  -88., 191., 103.],
       [-360., -184., 375., 199.],
       [ -56.,  -56.,  71.,  71.],
       [-120., -120., 135., 135.],
       [-248., -248., 263., 263.],
       [ -36.,  -80.,  51.,  95.],
       [ -80., -168.,  95., 183.],
       [-168., -344., 183., 359.]]
    ))

    y = keras_rcnn.backend.anchor()
    tensorflow.assert_equal(x, y)
    numpy.testing.assert_array_equal(x, y)


def test_clip():
    pass


def test_shift():
    x = (1764, 4)

    y = keras_rcnn.backend.shift((14, 14), 16).shape

    assert x == y


def test_inside_image():
    stride = 16
    features = (14, 14)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    img_info = (224, 224, 1)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info)

    assert inds_inside.shape == (84,)

    assert all_inside_anchors.shape == (84, 4)


def test_overlapping():
    stride = 16
    features = (14, 14)
    img_info = (224, 224, 1)
    gt_boxes = tensorflow.zeros((91, 4))

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info)

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = keras_rcnn.backend.overlapping(gt_boxes, all_inside_anchors, inds_inside)

    assert argmax_overlaps_inds.shape == (84, )

    assert max_overlaps.shape == (84, )

    assert gt_argmax_overlaps_inds.shape == (91, )


def test_label():
    stride = 16
    feat_h, feat_w = (14, 14)
    img_info = (224, 224, 1)

    gt_boxes = tensorflow.zeros((91, 4))

    all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), stride)

    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(all_bbox, img_info)

    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(gt_boxes, all_inside_bbox, inds_inside)

    assert argmax_overlaps_inds.shape == (84, )

    assert bbox_labels.shape == (84, )

'''
def test_regression(anchor_layer, gt_boxes):
    _, y_true, inds_inside, _ = anchor_layer.call(gt_boxes)

    y_true = keras.backend.eval(y_true)

    # anchor_layer.regression(y_true, y_pred, inds_inside)

    assert True


def test_compute_output_shape(anchor_layer):
    assert anchor_layer.compute_output_shape((14, 14)) == (None, None, 4)
'''
