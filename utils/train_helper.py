from graphs.models.DeepLabV2_feat import DeeplabFeat

def get_model(args):
    if args.backbone == "resnet101":
        model = DeeplabFeat(num_classes=args.num_classes, backbone='ResNet101', pretrained=args.imagenet_pretrained)
    elif args.backbone == "vgg16":
        model = DeeplabFeat(num_classes=args.num_classes, backbone='VGG16', pretrained=args.imagenet_pretrained)
    else:
        raise ValueError('{} segmentation network is not allowed, choose from: resnet101 or vgg16'.format(args.backbone))

    params = model.optim_parameters(args)
    args.numpy_transform = True
    return model, params