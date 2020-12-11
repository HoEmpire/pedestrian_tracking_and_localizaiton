import baseline
import reid_config


def build_model():
    num_classes = 1
    last_stride = 1
    cfg = reid_config.Config()
    pretrain_path = cfg.weights_path
    # pretrain_path = '/home/tim/duke_resnet50_model_120_rank1_864.pth'
    model_neck = 'bnneck'
    neck_feat = "after"
    model_name = "resnet50"
    pretrain_choice = "self"
    model = baseline.Baseline(num_classes, last_stride, pretrain_path,
                              model_neck, neck_feat, model_name,
                              pretrain_choice)
    model.load_param(pretrain_path)
    model.eval()
    model.cuda()
    return model