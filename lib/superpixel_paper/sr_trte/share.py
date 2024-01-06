
def config_via_spa(cfg):
    spa = cfg.spa_version
    if spa == "espa":
        cfg.topk = -1
        cfg.intra_version = "v2"
    elif spa == "easpa":
        cfg.intra_version = "v2"
    elif spa == "aspa":
        cfg.intra_version = "v1"
    elif spa == "nat":
        cfg.use_intra = False
        cfg.use_nat = True
    else:
        cfg.use_intra = False
        cfg.use_conv = True

