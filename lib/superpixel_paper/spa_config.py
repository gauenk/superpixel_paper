
def config_via_spa(cfg,verbose=True):
    def vprint(*args,**kwargs):
        if verbose: print(*args,**kwargs)

    spa = cfg.spa_version
    if spa == "slic_mle":
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":"ones",
                 "spa_sim_method":"slic"}
        for key,val in pairs.items():
            if key in cfg:
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa in ["nsa_mle","sna"]:
        pairs = {"spa_attn_normz":"mle",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":None,
                 "spa_sim_method":"slic",
                 "gen_sp_use_grad":"detach_x",
                 "use_sna":True,
                 "use_nsp":True,
                 "use_spa":False}
        for key,val in pairs.items():
            if key in cfg:
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                 "use_sna":True,
                 "use_nsp":True,
                 "use_spa":False}
        for key,val in pairs.items():
            if key in cfg:
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_km: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("slic_mle: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "ssna":
        pairs = {"use_ssna":True,
                 # "use_intra":False,
                 "use_spa":False,
                 "use_nat":False,
                 "use_conv":False,
                 "gen_sp_type":"reshape"}
        for key,val in pairs.items():
            if key in cfg:
                vprint(": not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("bass_mle: not setting %s:%s" %(str(key),str(cfg[key])))
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
                vprint("bass_s: not setting %s:%s" %(str(key),str(cfg[key])))
                continue
            cfg[key] = val
    elif spa == "aspa":
        pairs = {"spa_attn_normz":"softmax",
                 "spa_vweight":False,
                 "spa_oweight":False,
                 "spa_scatter_normz":"ones"}
        for key,val in pairs.items():
            if key in cfg:
                vprint("aspa: not setting %s:%s" %(str(key),str(cfg[key])))
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
    # elif spa == "sna":
    #     cfg.use_intra = False
    #     cfg.use_spa = False
    #     cfg.use_nsp = True
    #     cfg.use_sna = True
    elif spa == "nsp":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_nsp = True
        cfg.use_sna = True
    elif spa == "conv":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_conv = True
    elif spa == "dncnn":
        cfg.use_intra = False
        cfg.use_spa = False
        cfg.use_conv = False
        cfg.use_dncnn = True
    else:
        raise ValueError(f"Uknown SPA version [{spa}]")
