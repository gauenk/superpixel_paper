

psize = 50
nsamples = 10
pk_nthreads = 1

for index in range(100):
    si = index % nsamples
    tmp = (index // nsamples)
    pi = tmp % psize
    tmp = (index // (nsamples*psize))
    pj = tmp % psize
    tmp =  (index // (nsamples*psize*psize))
    pk_init = tmp % pk_nthreads;
    print(si,pi,pj,pk_init)
