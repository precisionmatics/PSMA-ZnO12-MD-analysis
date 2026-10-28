#!/usr/bin/env python3
"""index.ndx from solv_ions.gro: System Protein NAAG ZNO Metals Solute (non-solvent) Solvent (SOL,NA,CL)"""
import sys,collections
fn=sys.argv[1]; out=sys.argv[2]; ndry=int(sys.argv[4])   # ndry = atoms in dry.gro (protein, NAAG, cage, metals, crystal waters)
L=open(fn).read().splitlines(); n=int(L[1]); G=collections.OrderedDict((k,[]) for k in ["System","Protein","NAAG","ZNO","Metals","Solute","Solvent"])
for i,l in enumerate(L[2:2+n],1):
    rn=l[5:10].strip(); G["System"].append(i)
    if i<=ndry: G["Solute"].append(i)
    else: G["Solvent"].append(i)
    if rn in("ZN2","CAL"): G["Metals"].append(i)
    elif rn=="ZNO": G["ZNO"].append(i)
    elif rn in("ACE",) or int(l[:5])>=90000: pass
nprot=10946; nn=34 if len(sys.argv)>3 and sys.argv[3]=="naag" else 0
G["Protein"]=list(range(1,nprot+1)); G["NAAG"]=list(range(nprot+1,nprot+nn+1))
with open(out,"w") as f:
    for k,v in G.items():
        if not v: continue
        f.write("[ %s ]\n"%k)
        for s in range(0,len(v),15): f.write(" ".join(str(x) for x in v[s:s+15])+"\n")
print({k:len(v) for k,v in G.items()})
