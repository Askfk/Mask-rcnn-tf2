import collections

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from Project import utils_graph
from . import utils


import efficientnet.tfkeras as efn


##########################################################################
# Utility Functions
##########################################################################


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("Shape: {:20}   ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}   max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}   max: {:10}".format("", ""))

        text += "  {}".format(array.dtype)
    print(text)


class BatchNorm(keras.layers.BatchNormalization):
    """Extends the tensorflow.keras BatchNormalization class to allow a central place
        to make changes if needed.

        Batch normalization has a negative effect on training if batches are small
        so this layer is often frozen (via setting in Config class) and functions
        as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
        None: Train BN layers. This is the normal mode
        False: Freeze BN layers. Good when batch size is small
        True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


##########################################################################
# Basic Feature Extraction Network (Based on EfficientNet)
##########################################################################

# TODO: Build basic feature extraction networks,
#  properly consider to refer to efficientnet or MnasNet to build an efficientnet network
#  with less params and FLOPS









##########################################################################
# Region Proposal Network (RPN)
##########################################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """
    Builds the computation graph of Region Proposal Network.

    :param feature_map: Basic features [batch, height, width, depth]
    :param anchors_per_location: number of anchors per pixel in the feature map
    :param anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                          every pixel in the feature map), or 2 (every other pixel).
    :return: rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
             rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
             rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                       applied to anchors.
    """
    # TODO: Check if stride of 2 causes alignment issues if the feature map is not even

    # Shared convolutional base of the RPN
    shared = keras.layers.Conv2D(512,  # TODO: Need to be changed if the basic feature map channel is different
                                 (3, 3), padding='same', activation='relu', strides=anchor_stride,
                                 name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2]
    x = keras.layers.Conv2D(2 * anchors_per_location, (1, 1), padding='valid', activation='linear',
                            name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG
    rpn_probs = keras.layers.Activation("softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [Batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h]
    x = keras.layers.Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                            activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_loaction, depth):
    """Builds a tensorflow.Keras model of the Region Proposal Network.
        It wraps the RPN graph so it can be used multiple times with shared
        weights.

        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                       every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

        Returns a Keras Model object. The model outputs, when called, are:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                    applied to anchors.
    """
    input_feature_map = keras.layers.Input(shape=[None, None, depth],
                                           name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_loaction, anchor_stride)
    return keras.Model([input_feature_map], outputs, name='rpn_model')


##########################################################################
# Proposal Layer
##########################################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    centre_y = boxes[:, 0] + .5 * height
    centre_x = boxes[:, 1] + .5 * width
    # Apply deltas
    centre_y += deltas[:, 0] * height
    centre_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = centre_y - .5 * height
    x1 = centre_x - .5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name='apply_box_deltas_out')
    return result


def clips_box_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(keras.layers.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.math.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.IMAGES_PER_GPU)

        # Apply deltas to anchors to get refined anchors
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: apply_box_deltas_graph(x, y),
                                  self.config.IMAGES_PER_GPU, names=['refined_anchors'])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clips_box_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                                                   self.nms_threshold, name='rpn_non_max_suppression')
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return [None, self.proposal_count, 4]


############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2)


class PyramidROIAlign(keras.layers.Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
        TODO: Finish basic feature extraction network and then change codes here.
        Params:
        - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

        Inputs:
        - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                 coordinates. Possibly padded with zeros if not enough
                 boxes to fill the array.
        - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        - feature_maps: List of feature maps from different levels of the pyramid.
                        Each is [batch, height, width, channels]

        Output:
        Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = pool_shape

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # TODO(checked): Feature maps format may need to be changed.
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1

        # TODO: Check whether boxes pre-processing is correct.
        shape = tf.shape(boxes)
        boxes = tf.reshape([-1, -1, 4])
        # Generate box indices.
        box_indices0 = tf.zeros([tf.shape[boxes] // 2])
        box_indices1 = tf.ones_like(box_indices0)
        box_indices = tf.concat([box_indices0, box_indices1], axis=0)

        boxes = tf.stop_gradient(boxes)

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how it's done in tf.crop_and_resize()
        # Result: [batch * num_boxes, pool_height, pool_width, channels]
        pooled = tf.image.crop_and_resize(
            feature_maps, boxes, box_indices, self.pool_shape, method='bilinear')

        # TODO: Make sure whether if need to rearrange pooled features to match
        #  the order of original boxes.
        pooled = tf.reshape(pooled, [shape[:2], tf.shape(pooled)[1:]])
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )



