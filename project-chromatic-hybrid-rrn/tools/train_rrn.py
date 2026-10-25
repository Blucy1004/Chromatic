import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import chess
import torch
import torch.nn.functional as F

FEAT,HID,REL,STEPS=19,8,4,3

def xorshift(state):
    state ^= (state << 13)&0xffffffff; state ^= state>>17; state ^= (state<<5)&0xffffffff
    return state&0xffffffff

def small(state,scale):
    state=xorshift(state); centered=(state&0xffff)-32768
    return state, (centered/32768.0)*scale

def initial(seed=2026):
    state=seed or 0x9e3779b9
    def vals(n,s):
        nonlocal state
        out=[]
        for _ in range(n): state,v=small(state,s);out.append(v)
        return out
    inp=np.array(vals(FEAT*HID,.08),dtype=np.float32).reshape(FEAT,HID)
    ib=np.array(vals(HID,.015),dtype=np.float32)
    selfm=np.eye(HID,dtype=np.float32)*.20
    rel=np.zeros((REL,HID,HID),dtype=np.float32)
    for r in range(REL):
        rel[r]=np.eye(HID,dtype=np.float32)*(.025*(r+1))
        rel[r]+=np.array(vals(HID*HID,.006),dtype=np.float32).reshape(HID,HID)
    mb=np.array(vals(HID,.01),dtype=np.float32)
    rank=np.array(vals(2*HID+5,.05),dtype=np.float32)
    return inp,ib,selfm,rel,mb,rank,0.

def make_sample(row):
    b=chess.Board(row['fen']); features=np.zeros((64,FEAT),np.float32)
    for sq,p in b.piece_map().items():
        idx=(0 if p.color else 6)+(p.piece_type-1);features[sq,idx]=1
    features[:,12]=float(b.turn)
    features[:,13:17]=[float(b.has_kingside_castling_rights(chess.WHITE)),float(b.has_queenside_castling_rights(chess.WHITE)),float(b.has_kingside_castling_rights(chess.BLACK)),float(b.has_queenside_castling_rights(chess.BLACK))]
    if b.ep_square is not None: features[b.ep_square,17]=1
    features[:,18]=min(b.halfmove_clock,100)/100
    occ=np.zeros((REL,64,64),np.float32)
    pieces=[b.piece_at(i) for i in range(64)]
    for src in range(64):
      for tgt in range(64):
        if src==tgt: continue
        dx=abs(src%8-tgt%8);dy=abs(src//8-tgt//8)
        if dx<=1 and dy<=1: occ[0,tgt,src]=1
        if pieces[src] and pieces[tgt] and pieces[src].color==pieces[tgt].color:
          occ[1,tgt,src]=1
          if pieces[src].piece_type==pieces[tgt].piece_type:occ[2,tgt,src]=1
    moves=list(b.legal_moves);cand=occ[3]
    mfeat=[]
    for m in moves:
      cand[m.to_square,m.from_square]=1
      mover=b.piece_type_at(m.from_square);capt=b.piece_type_at(m.to_square)
      if mover==chess.PAWN and m.to_square==b.ep_square and capt is None and chess.square_file(m.from_square)!=chess.square_file(m.to_square):capt=chess.PAWN
      center=1-(abs(3-chess.square_file(m.to_square))+abs(3-chess.square_rank(m.to_square)))/6
      castle=mover==chess.KING and abs(m.to_square-m.from_square)==2
      mfeat.append([mover/6,(capt or 0)/5,(m.promotion or 0)/5,center,float(castle)])
    # Relation edge aggregate is divided by the incoming relation degree at each target.
    for r in range(REL):
      counts=occ[r].sum(axis=1,keepdims=True);np.divide(occ[r],np.maximum(counts,1),out=occ[r])
    label=chess.Move.from_uci(row['bestmove'])
    if label not in moves: raise ValueError(f'teacher label not legal: {row["fen"]} {label}')
    return {'x':torch.from_numpy(features),'a':torch.from_numpy(occ),'src':torch.tensor([m.from_square for m in moves]),'dst':torch.tensor([m.to_square for m in moves]),'mf':torch.tensor(mfeat,dtype=torch.float32),'label':moves.index(label),'fen':row['fen']}

class RRN(torch.nn.Module):
 def __init__(self,init):
  super().__init__();inp,ib,selfm,rel,mb,rank,rb=init
  self.inp=torch.nn.Parameter(torch.tensor(inp.copy()));self.ib=torch.nn.Parameter(torch.tensor(ib.copy()))
  self.selfm=torch.nn.Parameter(torch.tensor(selfm.copy()));self.rel=torch.nn.Parameter(torch.tensor(rel.copy()));self.mb=torch.nn.Parameter(torch.tensor(mb.copy()))
  self.rank=torch.nn.Parameter(torch.tensor(rank.copy()));self.rb=torch.nn.Parameter(torch.tensor(rb,dtype=torch.float32))
 def forward(self,s):
  h=torch.tanh(s['x']@self.inp+self.ib)
  for _ in range(STEPS):
   z=h@self.selfm.T+self.mb
   for r in range(REL): z=z+(s['a'][r]@h)@self.rel[r].T
   h=torch.tanh(z)
  f=torch.cat([h[s['src']],h[s['dst']],s['mf']],dim=1)
  return f@self.rank+self.rb

def metrics(model,data):
 model.eval();loss=0.;top=0;rr=0.
 with torch.no_grad():
  for s in data:
   scores=model(s); target=s['label'];loss+=float(F.cross_entropy(scores[None,:],torch.tensor([target])))
   order=torch.argsort(scores,descending=True);top+=int(int(order[0])==target);rr+=1/(int((order==target).nonzero()[0])+1)
 return {'loss':loss/len(data),'top1':top/len(data),'mrr':rr/len(data)}

def export(model,path):
 vals=lambda x:x.detach().cpu().numpy().reshape(-1).tolist()
 inp=model.inp.detach().cpu().numpy().reshape(-1);ib=vals(model.ib);selfm=vals(model.selfm);rel=vals(model.rel);mb=vals(model.mb);rank=vals(model.rank);rb=float(model.rb.detach().cpu())
 with open(path,'w',encoding='ascii',newline='\n') as f:
  f.write(f'CHROMATIC_RRN_WEIGHTS 1 {FEAT} {HID} {REL} {STEPS}\n')
  f.write('learned 1 seed 20261025 provenance teacher-sf19-n1024-32traj-v1\n')
  for name,arr in [('input',inp),('input_bias',ib),('self',selfm),('relation',rel),('message_bias',mb),('rank',rank)]:f.write(name+' '+' '.join(format(float(v),'.9g') for v in arr)+'\n')
  f.write('rank_bias '+format(rb,'.9g')+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=200);p.add_argument('--wall-seconds',type=float,default=120);a=p.parse_args()
 torch.manual_seed(20261025);np.random.seed(20261025);random.seed(20261025);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 data=json.loads(Path(a.data).read_text(encoding='utf-8'));tr=[make_sample(x) for x in data['train']];va=[make_sample(x) for x in data['heldout']]
 model=RRN(initial());base_train=metrics(model,tr);base_val=metrics(model,va)
 opt=torch.optim.Adam(model.parameters(),lr=.001);rng=random.Random(20261025);begin=time.monotonic();history=[];done=0
 model.train()
 for step in range(a.steps):
  if time.monotonic()-begin>=a.wall_seconds:break
  picks=[tr[rng.randrange(len(tr))] for _ in range(32)];opt.zero_grad();losses=[]
  for s in picks:
   scores=model(s);losses.append(F.cross_entropy(scores[None,:],torch.tensor([s['label']])))
  loss=torch.stack(losses).mean();loss.backward();opt.step();done+=1
  if done%25==0:history.append({'step':done,'loss':float(loss.detach()),'elapsed':time.monotonic()-begin})
 final_train=metrics(model,tr);final_val=metrics(model,va)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);export(model,out)
 report={'seed':20261025,'optimizer':'Adam','lr':.001,'batch':32,'threads':2,'requested_steps':a.steps,'completed_steps':done,'train_seconds':time.monotonic()-begin,'train_samples':len(tr),'heldout_samples':len(va),'reference_train':base_train,'reference_heldout':base_val,'candidate_train':final_train,'candidate_heldout':final_val,'history':history,'weight_path':str(out)}
 out.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
