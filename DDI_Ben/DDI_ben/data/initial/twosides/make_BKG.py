import json
from tqdm import tqdm
f0 = open('data/hetionet-v1.0-edges.sif','r')
f1 = open('data/cid2db.txt','r')
f2 = open('cid2id.json','r')
f2j = json.load(f2)
id2allname={}
rel2id={}
for cid in f2j.keys():
    id2allname[len(id2allname)]=cid

for line in f1:
    msg = line.strip().split()
    cid = 'CID'+msg[0].zfill(9)
    if cid in f2j.keys() and len(msg)>1:
        dbname = 'Compound::'+msg[1]
        id2allname[f2j[cid]]=dbname

allname2id=dict([val,key] for key,val in id2allname.items())

f3 = open('data/hetionet-v1.0-nodes.tsv','r')
for line in f3:
    msg = line.strip().split()
    if msg[0]=='id':
        continue
    if  msg[0] in allname2id:
        continue
    else:
        allname2id[msg[0]]=len(allname2id)

# f4 = open('allname2id.json','w')
# f4.write(json.dumps(allname2id))

f5 = open('data/relation2id.json','r')
relation2id = json.load(f5)
f6 = open('data/hetionet-v1.0-edges.sif','r')
f7 = open('new_data/data.txt','r')
f8 = open('new_BKG_2.txt','w')
dataset = []
for line in f7:
    h,t,r,p = line.strip().split('\t')
    if p =='1':
        dataset.append((h,t))

for line in tqdm(f6):
    h,r,t = line.strip().split('\t')
    if h == 'source' or r == 'GpBP':
        continue
    if h not in allname2id:
        allname2id[h] = len(allname2id)
    if t not in allname2id:
        allname2id[t] = len(allname2id)
    hid = allname2id[h]
    rid = relation2id[r]
    tid = allname2id[t]
    # if hid ==163 or rid == 163:
    #     xxxx = 0
    if hid<645 and rid<645 and (hid,tid) in dataset:
        continue
    else:
        msg = str(hid)+' '+str(tid)+' '+str(rid)+'\n'
        f8.write(msg)

f4 = open('allname2id_2.json','w')
f4.write(json.dumps(allname2id))