from .graph_ssm import GraphSSM


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "convolutional_graph_ssm":
        model = GraphSSM(
            num_classes=config.MODEL.NUM_CLASSES,
            channels=config.MODEL.CONV_GRAPH_SSM.CHANNELS,
            depths=config.MODEL.CONV_GRAPH_SSM.DEPTHS,
            layer_scale=config.MODEL.CONV_GRAPH_SSM.LAYER_SCALE,
            post_norm=config.MODEL.CONV_GRAPH_SSM.POST_NORM,
            mlp_ratio=config.MODEL.CONV_GRAPH_SSM.MLP_RATIO,
            with_cp=config.TRAIN.USE_CHECKPOINT,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
