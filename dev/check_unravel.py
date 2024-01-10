

psize = 3
nsamples = 3
ntotal = nsamples*psize*psize*psize
for index in range(ntotal):
    si = index % nsamples
    tmp = (index // nsamples)
    pi = tmp % psize
    tmp = (index // (nsamples*psize))
    pj = tmp % psize
    tmp =  (index // (nsamples*psize*psize))
    pk = tmp % psize;
    print(si,pi,pj,pk)
