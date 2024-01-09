
def config_via_spa(cfg):
    spa = cfg.spa_version
    if spa == "espa":
        cfg.topk = -1
        cfg.intra_version = "v2"
    elif spa == "easpa":
        cfg.intra_version = "v2"
    elif spa == "exact":
        cfg.intra_version = "v3"
        # -- optional --
        pairs = {"spa2_normz":"mask",
                 "spa2_kweight":False,
                 "spa2_oweight":False}
        for key,val in pairs.items():
            if key in cfg: continue
            cfg[key] = val
    elif spa == "expectation":
        pairs = {"spa2_normz":None,
                 "spa2_kweight":True,
                 "spa2_oweight":True}
        cfg.intra_version = "v4"
    elif spa == "flex":
        cfg.intra_version = "v2"
    elif spa == "aspa":
        cfg.intra_version = "v1"
    elif spa == "nat":
        cfg.use_intra = False
        cfg.use_nat = True
    elif spa == "conv":
        cfg.use_intra = False
        cfg.use_conv = True
    else:
        raise ValueError(f"Uknown SPA version [{spa}]")
