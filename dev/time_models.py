import time
import copy
import cache_io
import torch
import torch as th
dcopy = copy.deepcopy
from superpixel_paper.deno_trte import sr_utils as utils

## Define hook functions
take_time_dict = {}

def take_time_pre(layer_name,module, input):
    take_time_dict[layer_name] = time.time()

def take_time(layer_name,module, input, output):
    take_time_dict[layer_name] =  time.time() - take_time_dict[layer_name]
    ## for TensorBoard you should use writter

from functools import partial

# Create Model
exp_fn = "exps/trte_deno/train.cfg"
cache_fn =".cache_io_exps/trte_deno/train/"
exps,uuids = cache_io.train_stages.run(exp_fn,cache_fn,update=True)
cfg = dcopy(exps[0])
# cfg = dcopy(exps[1])

import_str = 'superpixel_paper.nlrn.model'
device = "cuda:0"
model = utils.import_module(import_str).create_model(cfg).to(device)
x = torch.rand(1,3,96,96).to(device)
model(x)


# Register function for every
for name,layer in model.blocks.named_children():
    for _name,_layer in layer.named_children():
        # print(name)
        if "nl_block" in _name:
            lname = "%s_%s"%(name,_name)
            _layer.register_forward_pre_hook( partial(take_time_pre, lname) )
            _layer.register_forward_hook( partial(take_time, lname) )
            # for name1,layer1 in _layer.named_children():
            #     lname = "%s_%s_%s"%(name,_name,name1)
            #     layer1.register_forward_pre_hook( partial(take_time_pre, lname) )
            #     layer1.register_forward_hook( partial(take_time, lname) )
x = torch.rand(10,3,96,96).to(device)
model(x)
print(take_time_dict)


print(model.block0.ssna)
