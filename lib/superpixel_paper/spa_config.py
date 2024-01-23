
def config_via_spa(cfg):
    spa = cfg.spa_version
    if spa == "slic_mle":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":"ones",
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "nsa_mle":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic",
                 "use_nsp":True,
                 "use_spa":False}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "nsa_mle_s":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic",
                 "spa_full_sampling":True,
                 "spa_attn_nsamples":3,
                 "spa_scatter_normz":"ones",
                 "use_nsp":True,
                 "use_spa":False}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_s":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_full_sampling":True,
                 "spa_attn_nsamples":10,
                 "spa_scatter_normz":"ones",
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_km":
        pairs = {"spa_attn_normz":"mle_z",
                 # "spa_attn_normz":"mle",
                 "spa_vweight":True,
                 "spa_oweight":True,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_km: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_ks":
        pairs = {"spa_attn_normz":"sample",
                 "spa_vweight":True,
                 "spa_oweight":True,
                 "spa_attn_normz_nsamples":10,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_biased": # replace attn normalization with standard softmax
        pairs = {"spa_attn_normz":"softmax",
                 "spa_vweight":True,
                 "spa_oweight":True,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_biased_v": # replace attn normalization with standard softmax
        pairs = {"spa_attn_normz":"softmax",
                 "spa_vweight":True,
                 "spa_oweight":True,
                 "spa_scatter_normz":"sum2one",
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "slic_biased_c": # replace attn normalization with standard softmax
        pairs = {"spa_attn_normz":"softmax",
                 "spa_vweight":True,
                 "spa_oweight":True,
                 "spa_scatter_normz":"ones",
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                print("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "bass_mle":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"bass"}
        for key,val in pairs.items():
            if key in cfg:
                print("bass_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "bass_s":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_full_sampling":True,
                 "spa_attn_nsamples":10,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"bass"}
        for key,val in pairs.items():
            if key in cfg:
                print("bass_s: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "aspa":
        pairs = {"spa_attn_normz":"softmax",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":"ones"}
        for key,val in pairs.items():
            if key in cfg:
                print("aspa: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "nat":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_nat = True
    elif spa == "none":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_nat = False
    elif spa == "nsp":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_nsp = True
    elif spa == "conv":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_conv = True
    else:
        raise ValueError(f"Uknown SPA version [{spa}]")
