"""Build FPN heads."""


import tensorflow as tf
from tensorflow import keras
from Project.ROIAlignLayer import ROIAlign
from Project.BatchNorm import BatchNormLayer


def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    # TODO: fc_layer_size may need to be changed to reduce params
    """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.

        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

        Returns:
            logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
            probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
            bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                         proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = ROIAlign([pool_size, pool_size],
                 name="roi_align_classifier")([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size,
                                                         (pool_size, pool_size),
                                                         padding='valid'),
                                     name='mrcnn_class_conv1')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(fc_layers_size, (1, 1)),
                                     name='mrcnn_class_conv2')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    shared = keras.layers.Lambda(lambda f: keras.backend.squeeze(
        keras.backend.squeeze(f, 3), 2), name='pool_squeeze')(x)

    # Classifier head
    mrcnn_class_logits = keras.layers.TimeDistributed(
        keras.layers.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = keras.layers.TimeDistributed(
        keras.layers.Activation('softmax'), name='mrcnn_class')(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw)]
    x = keras.layers.TimeDistributed(
        keras.layers.Dense(num_classes * 4, activation='linear'),
        name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw)]
    s = keras.backend.int_shape(x)

    if s[1] == None:
        mrcnn_bbox = keras.layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = keras.layers.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

        rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from different layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        image_meta: [batch, (meta data)] Image details. See compose_image_meta()
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results
        train_bn: Boolean. Train or freeze Batch Norm layers

        Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = ROIAlign([pool_size, pool_size],
                 name='roi_align_mask')([rois, image_meta] + feature_maps)

    # Conv layers
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same'),
                                     name='mrcnn_mask_conv1')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(),
                                     name='mask_rcnn_bn1')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same'),
                                     name='mrcnn_mask_conv2')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(),
                                     name='mrcnn_mask_bn2')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same'),
                                     name='mrcnn_mask_conv3')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(),
                                     name='mrcnn_mask_bn3')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding='same'),
                                     name='mrcnn_mask_conv4')(x)
    x = keras.layers.TimeDistributed(BatchNormLayer(),
                                     name='mrcnn_mask_bn4')(x, training=train_bn)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation='relu'),
                                     name='mrcnn_mask_deconv')(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'),
                                     name='mrcnn_mask')(x)
    return x
