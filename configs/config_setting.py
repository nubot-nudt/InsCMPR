import datetime
now = datetime.datetime.now()

class setting_config:
    """
    the config of training setting.
    """
    dataset_toot = "/home/shorwin/work/SemanticKitti/dataset"
    log_dir = "logs/"+now.strftime("%Y%m%d%H%M")+"flow/"
    nmf = False
    mlp = False
    lr = 0.0001
    resume = True
    pth = "./logs/202408270146flow/200_CMPR.pth.tar"
    spth = "CMPR.pth.tar"
    margin = 0.4
    mid_margin = 0.04
    resize_shape = (400*2, 64*2)
    Batch_Size =1
    num_works = 16
    K = 16
    epoches = 301
    