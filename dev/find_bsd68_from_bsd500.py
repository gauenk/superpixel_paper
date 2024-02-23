from PIL import Image
import numpy as np
from pathlib import Path

paths = [str(p).split("/")[-1] for p in Path("/home/gauenk/Documents/data/bsd500/images/all/").iterdir()]
fn_list_bsd500 = sorted(paths)
paths = [str(p).split("/")[-1] for p in Path("/home/gauenk/Documents/data/cbsd68/images/").iterdir()]
fn_list_bsd68 = sorted(paths)


bsd500_for_train = []
for fn_b500 in fn_list_bsd500:
    if fn_b500 in fn_list_bsd68:
        continue
    bsd500_for_train.append(fn_b500.split(".")[0])
print(bsd500_for_train)
fn_list_bsd68 = [f.split(".")[0] for f in fn_list_bsd68]
print(fn_list_bsd68)
print(len(bsd500_for_train),len(fn_list_bsd68))
with open("train_list.txt", "w") as fout:
    print(*bsd500_for_train, sep="\n", file=fout)
with open("test_list.txt", "w") as fout:
    print(*fn_list_bsd68, sep="\n", file=fout)

# np.savetxt("train_list.txt",np.array())
# np.savetxt("test_list.txt",np.array())

# bsd68_map = []
# for fn_b500 in fn_list_bsd500:
#     for fn_b68 in fn_list_bsd68:
#         img_b500 = np.array(Image.open(fn_b500))
#         img_b68 = np.array(Image.open(fn_b68).convert("RGB"))
#         if img_b500.shape[:2] != img_b68.shape[:2]: continue
#         delta = np.mean((img_b500 - img_b68)**2)
#         # print(img_b500.shape,img_b68.shape)
#         if delta < 1e-10:
#             # print("hi.")
#             bsd68_map.append([fn_b68,fn_b500])
#             break
# print(bsd68_map)
# print(len(bsd68_map))
# np.array(bsd68_map).savetxt("bsd68_test_list.txt")

# bsd500_for_training = []
# bsd68_map = np.array(bsd68_map)
# for fn_b500 in fn_list_bsd500:
#     if fn_b500 in bsd68_map[:,1]:
#         continue
#     bsd500_for_training.append(fn_b500)
# np.array(bsd500_for_training).savetxt("bsd500_train_list.txt")

