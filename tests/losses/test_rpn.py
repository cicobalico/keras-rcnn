import keras.backend
import keras.utils
import numpy
import keras_rcnn.losses.rpn


def test_classification():
    n_anchors = 9
    rpn_cls = keras_rcnn.losses.rpn._classification(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 2 * n_anchors)))
    expected_loss = - numpy.log(0.5)
    loss = keras.backend.eval(rpn_cls(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)


def test_regression():
    n_anchors = 9
    rpn_reg = keras_rcnn.losses.rpn._regression(n_anchors)
    y_pred = keras.backend.variable(0.5 * numpy.ones((1, 4, 4, 4 * n_anchors)))
    y_true = keras.backend.variable(numpy.ones((1, 4, 4, 8 * n_anchors)))
    expected_loss = numpy.power(0.5, 3)
    loss = keras.backend.eval(rpn_reg(y_true, y_pred))
    assert numpy.isclose(expected_loss, loss)


def test_encode():
    anchors = 9
    features = (14, 14)
    image_shape = (224, 224)
    samples = 91

    y_true = numpy.random.choice(range(0, image_shape[0]), 4*samples)
    y_true = y_true.reshape((-1, 4))
    y_true = numpy.expand_dims(y_true, 0)

    bbox_labels, bbox_reg_targets, inds_inside, n_all_bbox = keras_rcnn.losses.rpn.encode(features, image_shape, y_true)

    assert bbox_labels.shape == (84, )

    assert bbox_reg_targets.shape == (84, 4) #keras.backend.int_shape(bbox_reg_targets) == (84, 4)

    assert inds_inside.shape == (84, )

    assert n_all_bbox == features[0]*features[1]*anchors


def test_proposal():
    anchors = 9
    features = (14, 14)
    image_shape = (224, 224)
    stride = 16

    y_pred_classification = numpy.zeros((1, features[0], features[1], anchors))
    y_pred_regression = numpy.zeros((1, features[0], features[1], anchors * 4))

    y_pred = numpy.concatenate([y_pred_regression, y_pred_classification], -1)

    y_true = numpy.zeros((100, 4))
    y_true = numpy.expand_dims(y_true, 0)
    loss = keras_rcnn.losses.rpn.proposal(anchors, image_shape=image_shape, stride=stride)(y_true, y_pred)

    assert keras.backend.eval(loss) == 0.0
