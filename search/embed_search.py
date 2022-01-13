#preprocess, gene embed vec
from transformers import  AlbertForMaskedLM
import os
#data process
from torch.utils.data import Dataset
from transformers import  AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
# preprocess
tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_small",local_files_only=True)
albert = torch.load('D:\\Z_DATA\\u4420199\\site_alb\\search\\res_alb.pkl')
class Seqset(Dataset):
    def __init__(self,seq,label):
        super().__init__()
        self.seq=[]
        for inputtext in seq:
            res=torch.tensor(tokenizer.encode(inputtext,add_special_tokens=True)).unsqueeze(0)
            res= F.pad(input= res,pad=(0,  128-res.shape[1],0, 0), mode='constant', value=0)
            self.seq.append(res)

        self.label=label
    def __getitem__(self, item):
        seq=self.seq[item]
        label=self.label[item]
        return seq,label
    def __len__(self):
        return len(self.seq)
'''
data=[]
datapath='./dataset/'
for i in os.listdir(datapath):
    file=open(datapath+i,'r',encoding='utf-8').read().split('\n')
    for line in file:
        data.append(line)
data=[sample.split(' ') for sample in data]
seq=[sample[0] for sample in data if len(sample)==2]
label=[int(sample[1]) for sample in data if len(sample)==2]
# train_seq,test_seq,train_label,test_label=train_test_split(seq,label,test_size=0.2)
#set config


search_data=Seqset(seq,label)

batch_size=256
class_num=3
emb_len=128

search_loader=DataLoader(search_data,
                          batch_size=batch_size, shuffle=False,
                          num_workers=0, drop_last=True)

albert=albert.cuda()
# albert.load_state_dict(torch.load('res.pkl'))

emb=[]
with torch.no_grad():
    for i, data in enumerate(search_loader):
        seq = data[0].view(batch_size, emb_len).cuda()
        label = torch.zeros(batch_size, class_num). \
            scatter_(1, data[1].view(batch_size, 1) - 1, 1).cuda()
        res = albert(input_ids=seq, labels=label)
        emb.append(res.hidden_states.cpu())
        if i%10==0:
            print('fin',i)
emb=torch.cat(emb,dim=0)
import pickle
pickle.dump(emb,open('emb.pkl','wb'))
'''

database=pickle.load(open('D:\\Z_DATA\\u4420199\\site_alb\\search\\emb_alb.pkl', 'rb')).numpy()

from sklearn.neighbors import KDTree
tree=KDTree(database)
print('process fin')

# emb_search
lines=[]
path='D:\\Z_DATA\\u4420199\\site_alb\\data\\'
for i in os.listdir(path):
    f=open(path+i,'r',encoding='utf-8').read().split('\n')
    for i in f:
        lines.append(i)
sample="明星 回应 爱国 问题 标准 来"
sample="明星回应爱国问题标准来"
sample="明星回应爱国问题标准"

stopwords=open(os.getcwd()+'/stop.txt','r',encoding='utf-8').read().split('\n')
check_stop=set(stopwords)

def search(query):
    res = torch.tensor(tokenizer.encode(query, add_special_tokens=True)).unsqueeze(0)
    seq = F.pad(input=res, pad=(0, 128 - res.shape[1], 0, 0), mode='constant', value=0)
    label=torch.tensor([0])# no use, just input
    with torch.no_grad():
        emb=albert(input_ids=seq, labels=label)
        emb=emb.hidden_states.cpu()
    nearest_dist, nearest_ind = tree.query(emb, k=11)
    nearest_ind=nearest_ind[0]
    nearest_dist=nearest_dist[0]
    print('query',query)
    print(nearest_ind)
    print(nearest_ind.dtype)
    ans=[{"score": nearest_dist[i], "text": lines[nearest_ind[i]]} for i in range(len(nearest_ind))]
    if abs(ans[0]['score']-0)<1e-6 or query in check_stop:
        ans[0]={"score": 0, "text": 'query OOV'}
        # ans[1]={"score": 0, "text": 'query OOV'}
    return ans
    #print(nearest_dist)
    #print(nearest_ind)
    #for i in nearest_ind[0]:
    #    print(lines[i])
#emb_search(sample)
