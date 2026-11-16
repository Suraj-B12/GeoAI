import numpy as np, sys
d=np.load(sys.argv[1]); f=d["f"]; P=d["psd"]; v=d["v"]; G=d["gyro"]
band=(f>=0.5)&(f<=15)
print("windows",len(v),"speed pct",np.percentile(v,[5,25,50,75,95]).round(2))
norm=P/P[:,band].sum(1,keepdims=True)
fb=[0.4,0.6,0.8,1,1.3,1.6,2,2.5,3,4,5,6,8,10,12,15,20,25,30,40,60]
for lo,hi in [(2,3.5),(3.5,5),(5,6.5),(6.5,8),(8,12)]:
    m=(v>=lo)&(v<hi)
    med=np.exp(np.log(norm[m]+1e-12).mean(0))
    row=[np.log10(med[(f>=a)&(f<b)].mean()) for a,b in zip(fb[:-1],fb[1:])]
    print(f"v {lo}-{hi} n={m.sum():5d} "+" ".join(f"{x:5.2f}" for x in row))
print("bins"," ".join(f"{a:5.1f}" for a in fb[:-1]))
