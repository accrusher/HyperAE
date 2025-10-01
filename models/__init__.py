from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder

    activation = args.activation

    num_features = args.num_features
    num_dec_layers = args.num_dec_layers

    decoder_AH_type = args.decoder_AH_type
    loss_DEG_A_para = args.loss_DEG_A_para
    loss_DEG_Z_para = args.loss_DEG_Z_para
    loss_DEG_H_para = args.loss_DEG_H_para#HyperGragh

    loss_APA_A2H_para = args.loss_APA_A2H_para
    loss_APA_H2A_para = args.loss_APA_H2A_para#HyperGraph

    #alpha = args.alpha

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        norm=norm,
        decoder_AH_type=decoder_AH_type,
        loss_DEG_A_para=loss_DEG_A_para,
        loss_DEG_Z_para=loss_DEG_Z_para,
        loss_DEG_H_para=loss_DEG_H_para,#HyperGraph
        loss_APA_A2H_para=loss_APA_A2H_para,
        loss_APA_H2A_para=loss_APA_H2A_para,#HyperGraph
        #alpha=alpha
    )
    return model
