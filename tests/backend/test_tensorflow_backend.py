import numpy
import keras.backend
import keras_rcnn.backend
import tensorflow 

def test_label():
    stride = 16
    feat_h, feat_w = (14, 14)
    img_info = (224, 224, 1)

    gt_boxes = numpy.zeros((91, 4)) #tensorflow.zeros((91, 4))

    all_bbox = keras_rcnn.backend.shift((feat_h, feat_w), stride)

    inds_inside, all_inside_bbox = keras_rcnn.backend.inside_image(all_bbox, img_info)

    argmax_overlaps_inds, bbox_labels = keras_rcnn.backend.label(gt_boxes, all_inside_bbox, inds_inside)

    assert argmax_overlaps_inds.shape == (84, )

    assert bbox_labels.shape == (84, )


def test_shift():
    x = (1764, 4)

    y = keras_rcnn.backend.shift((14, 14), 16).shape

    assert x == y #keras.backend.eval(y)


def test_inside_image():
    stride = 16
    features = (14, 14)

    all_anchors = keras_rcnn.backend.shift(features, stride)

    img_info = (224, 224, 1)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info)

    assert inds_inside.shape == (84,) #keras.backend.eval(inds_inside.shape) == (84,)

    assert all_inside_anchors.shape == (84,4) # keras.backend.eval(all_inside_anchors).shape == (84, 4)


def test_overlapping():
    stride = 16
    features = (14, 14)
    img_info = (224, 224, 1)
    gt_boxes = numpy.zeros((91, 4)) #tensorflow.zeros((91, 4))

    all_anchors = keras_rcnn.backend.shift(features, stride)

    inds_inside, all_inside_anchors = keras_rcnn.backend.inside_image(all_anchors, img_info)

    argmax_overlaps_inds, max_overlaps, gt_argmax_overlaps_inds = keras_rcnn.backend.overlapping(gt_boxes, all_inside_anchors, inds_inside)

    assert argmax_overlaps_inds.shape == (84, )

    assert max_overlaps.shape == (84, )

    assert gt_argmax_overlaps_inds.shape == (91, )


def test_crop_and_resize():
    image = keras.backend.variable(numpy.ones((1, 28, 28, 3)))
    boxes = keras.backend.variable(numpy.array([[[0.1, 0.1, 0.2, 0.2],[0.5, 0.5, 0.8, 0.8]]]))
    size = [7, 7]
    slices = keras_rcnn.backend.crop_and_resize(image, boxes, size)
    assert keras.backend.eval(slices).shape == (2, 7, 7, 3)


def test_overlap():
    x = numpy.asarray([
        [0, 10, 0, 10],
        [0, 20, 0, 20],
        [0, 30, 0, 30],
        [0, 40, 0, 40],
        [0, 50, 0, 50],
        [0, 60, 0, 60],
        [0, 70, 0, 70],
        [0, 80, 0, 80],
        [0, 90, 0, 90]
    ])

    y = numpy.asarray([
        [0, 20, 0, 20],
        [0, 40, 0, 40],
        [0, 60, 0, 60],
        [0, 80, 0, 80]
    ])

    overlapping = keras_rcnn.backend.overlap(x, y)

    expected = numpy.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    numpy.testing.assert_array_equal(expected, overlapping)

