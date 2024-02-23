
def optional(pydict,key,default):
    if key in pydict: return pydict[key]
    else: return default

def extract(pydict,pairs):
    for key in pairs:
        pydict[key] = optional(pydict,key,pairs[key])

