from covid19.models import ResNet50, COVIDNet


def get_model(architecture, weights, n_classes, load_weights):
    if architecture == 'resnet50':
        model = ResNet50(n_classes, weights=weights)
    elif architecture == 'covidnet':
        model = COVIDNet(n_classes, weights=weights)
    else:
        raise ValueError('Invalid architecture')
    if load_weights is not None:
        model.load_weights(load_weights)
    return model


def get_image_size(architecture):
    if architecture == 'resnet50':
        return ResNet50.image_shape[0:2]
    elif architecture == 'covidnet':
        return COVIDNet.image_shape[0:2]
    else:
        raise ValueError('Invalid architecture')
