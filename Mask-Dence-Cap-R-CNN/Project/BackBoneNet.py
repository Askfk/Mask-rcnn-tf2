"""Build efficient bottom-up networks."""

import efficientnet.tfkeras as efn


def build_backbone_net_graph(input_tensor, architecture, weights=None):
    """
    Build basic feature extraction networks.
    :param input_tensor: Input of the basic networks, should be a tensor or tf.keras.layers.Input
    :param architecture: The architecture name of the basic network.
    :param weights: Whether download and initialize weights from the pre-trained weights,
                    could be either 'imagenet', (pre-training on ImageNet)
                                    'noisy-student',
                                    'None' (random initialization)，
                                    or the path to the weights file to be loaded。
    :return: Efficient Model and corresponding endpoints.
    """
    assert architecture in ['efficientnet-b0', 'efficientnet-b1',
                            'efficientnet-b2', 'efficientnet-b3',
                            'efficientnet-b4', 'efficientnet-b5',
                            'efficientnet-b7', 'efficientnet-b7',
                            'efficientnet-l2']

    if architecture == 'efficientnet-b0':
        return efn.EfficientNetB0(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b1':
        return efn.EfficientNetB1(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b2':
        return efn.EfficientNetB2(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b3':
        return efn.EfficientNetB3(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b4':
        return efn.EfficientNetB4(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b5':
        return efn.EfficientNetB5(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b6':
        return efn.EfficientNetB6(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-b7':
        return efn.EfficientNetB7(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    elif architecture == 'efficientnet-l2':
        return efn.EfficientNetL2(include_top=False, weights=weights,
                                  input_tensor=input_tensor,
                                  input_shape=[None, None, 3])
    else:
        raise ValueError("Argument architecture should in "
                         "[efficientnet-b0, efficientnet-b1, "
                         "efficientnet-b2, efficientnet-b3, efficientnet-b4, efficientnet-b5, "
                         "efficientnet-b7, efficientnet-b7, efficientnet-l2] "
                         "but get %s" % architecture)