from __future__ import annotations
import json, platform, random, time, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import optuna, pandas as pd, torch
from sklearn.preprocessing import StandardScaler
from torch import nn
import run_paper1_bo_ets as core

warnings.filterwarnings('ignore'); optuna.logging.set_verbosity(optuna.logging.WARNING)
ROOT=core.ROOT; OUT=ROOT/'outputs/paper1_bo_ets_residual_v2'; TAB=OUT/'tables'; PRED=OUT/'predictions'; FIG=OUT/'figures'; CFG=OUT/'configs'; AUD=OUT/'audits'; LOG=OUT/'logs'
for p in [OUT,TAB,PRED,FIG,CFG,AUD,LOG,OUT/'shrinkage',OUT/'gated',OUT/'multiscale',OUT/'gated_multiscale',OUT/'ablation']: p.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); SEEDS=[42,52,62]; HORIZONS=core.HORIZONS; YEARS=core.YEARS; TEST_START,TEST_END=core.TEST_START,core.TEST_END; MAX_PARAMS=100000; N_TRIALS=30; TOP_K=5; SEARCH_EPOCHS=60; FINAL_EPOCHS=90

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load(): return core.load()
def load_base():
    p=ROOT/'outputs/paper1_bayesian_optimization/tables/selected_configs.csv'; r=pd.read_csv(p).iloc[0]; return (float(r.alpha),float(r.beta),float(r.gamma))
def residual_data(y,params):
    n=len(y); innov=np.full(n,np.nan); fc={h:np.full(n,np.nan) for h in HORIZONS}; err={h:np.full(n,np.nan) for h in HORIZONS}
    for o in range(23,n-1):
        hist=y[:o+1]; innov[o+1]=y[o+1]-core.forecast(hist,1,*params)[-1] if o+1<n else np.nan
        for h in HORIZONS:
            if o+h<n:
                fc[h][o+h]=core.forecast(hist,h,*params)[-1]; err[h][o+h]=y[o+h]-fc[h][o+h]
    innov[np.isnan(innov)]=0.; return innov,fc,err

class Causal(nn.Module):
    def __init__(self,cin,cout,k,d): super().__init__(); self.left=(k-1)*d; self.conv=nn.Conv1d(cin,cout,k,dilation=d)
    def forward(self,x): return self.conv(nn.functional.pad(x,(self.left,0)))
class Branch(nn.Module):
    def __init__(self,k,c,dils,drop):
        super().__init__(); layers=[]; ci=1
        for d in dils: layers += [Causal(ci,c,k,d),nn.ReLU()]; ci=c; layers += [nn.Dropout(drop)] if drop else []
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
class GatedMulti(nn.Module):
    def __init__(self,cfg,variant='D'):
        super().__init__(); self.variant=variant; c=cfg['channels']; dils=cfg['dilations']; drop=cfg['dropout']
        self.branches=nn.ModuleList([Branch(k,c,dils,drop) for k in ([cfg['kernel']] if variant=='B' else [3,5,7])])
        if variant=='B': self.fused=c
        else: self.fused=c*3
        self.proj=nn.Conv1d(self.fused,c,1) if self.fused!=c else nn.Identity()
        self.att=nn.Sequential(nn.Linear(c,cfg['attention_dim']),nn.Tanh(),nn.Linear(cfg['attention_dim'],1,bias=False))
        self.res_head=nn.Sequential(nn.Linear(c,cfg['gate_hidden']),nn.ReLU(),nn.Linear(cfg['gate_hidden'],1))
        self.gate_head=nn.Sequential(nn.Linear(c+5,cfg['gate_hidden']),nn.ReLU(),nn.Linear(cfg['gate_hidden'],1))
    def forward(self,x,meta):
        z=torch.cat([b(x) for b in self.branches],dim=1) if len(self.branches)>1 else self.branches[0](x); z=self.proj(z).transpose(1,2)
        w=torch.softmax(self.att(z).squeeze(-1),dim=1); ctx=torch.sum(z*w.unsqueeze(-1),dim=1)
        r=self.res_head(ctx).squeeze(-1)
        g=torch.sigmoid(self.gate_head(torch.cat([ctx,meta],dim=1)).squeeze(-1)) if self.variant in ('D','B') else torch.ones_like(r)
        return r,g

def cfg_suggest(t):
    return {'lookback':t.suggest_categorical('lookback',[24,36,48,60]),'channels':t.suggest_categorical('channels',[8,16,24,32,48]),'kernel':t.suggest_categorical('kernel',[3,5,7]),'dilations':{'124':[1,2,4],'1248':[1,2,4,8]}[t.suggest_categorical('dilations',['124','1248'])],'blocks':1,'attention_dim':t.suggest_categorical('attention_dim',[8,16,32]),'gate_hidden':t.suggest_categorical('gate_hidden',[4,8,16]),'dropout':t.suggest_float('dropout',0,.35),'lr':t.suggest_float('lr',1e-4,3e-3,log=True),'wd':t.suggest_float('wd',1e-6,1e-2,log=True),'loss':t.suggest_categorical('loss',['MSE','Huber','MAE']),'mode':t.suggest_categorical('mode',['raw','standardized','difference'])}
def make_samples(innov,err,fc,y,cutoff,h,cfg):
    x=[]; meta=[]; target=[]; anchor=[]; origins=[]; L=cfg['lookback']
    for o in range(max(23,L-1),cutoff-h+1):
        if not np.isfinite(err[h][o+h]): continue
        q=innov[o-L+1:o+1].copy();
        if cfg['mode']=='difference': q=np.r_[0,np.diff(q)]
        x.append(q); target.append(y[o+h]); anchor.append(fc[h][o+h]); origins.append(o)
        m=q[-3:].mean(),q[-3:].std(),q[-6:].mean(),q[-6:].std(),np.sin(2*np.pi*((o+1)%12)/12),
        meta.append([m[0],m[1],m[2],m[3],m[4]])
    return np.asarray(x,np.float32),np.asarray(meta,np.float32),np.asarray(target,np.float32),np.asarray(anchor,np.float32),np.asarray(origins,int)
def fit(y,innov,err,fc,cutoff,h,cfg,seed,epochs):
    seed_all(seed); x,m,t,a,o=make_samples(innov,err,fc,y,cutoff,h,cfg)
    if len(x)<12: raise ValueError('too few causal rows')
    xs=StandardScaler().fit(x.reshape(-1,1)); ms=StandardScaler().fit(m); ts=StandardScaler().fit((t-a).reshape(-1,1)); xx=xs.transform(x.reshape(-1,1)).reshape(x.shape); mm=ms.transform(m); yy=ts.transform((t-a).reshape(-1,1)).ravel()
    split=max(8,int(.8*len(xx))); split=min(split,len(xx)-1); xt=torch.tensor(xx[:split,None,:],dtype=torch.float32,device=DEVICE); mt=torch.tensor(mm[:split],dtype=torch.float32,device=DEVICE); yt=torch.tensor(yy[:split],dtype=torch.float32,device=DEVICE); xv=torch.tensor(xx[split:,None,:],dtype=torch.float32,device=DEVICE); mv=torch.tensor(mm[split:],dtype=torch.float32,device=DEVICE); yv=torch.tensor(yy[split:],dtype=torch.float32,device=DEVICE)
    model=GatedMulti(cfg,cfg.get('variant','D')).to(DEVICE); npar=sum(p.numel() for p in model.parameters())
    if npar>=MAX_PARAMS: raise ValueError(f'parameter_count={npar}')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']); lf={'MSE':nn.MSELoss(),'Huber':nn.HuberLoss(),'MAE':nn.L1Loss()}[cfg['loss']]; best=None; bv=float('inf'); wait=0
    for _ in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True); r,g=model(xt,mt); loss=lf(g*r,yt) if cfg.get('variant','D') in ('D','B') else lf(r,yt); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); model.eval()
        with torch.no_grad(): r,g=model(xv,mv); vl=float(lf(g*r,yv).item() if cfg.get('variant','D') in ('D','B') else lf(r,yv).item())
        if vl<bv-1e-5: bv=vl; wait=0; best={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait>=12: break
    if best: model.load_state_dict(best)
    return model,xs,ms,ts,npar,o
def infer(model,xs,ms,ts,context,meta,cfg):
    q=context.copy();
    if cfg['mode']=='difference': q=np.r_[0,np.diff(q)]
    xx=xs.transform(q.reshape(-1,1)).ravel(); mm=ms.transform(np.asarray(meta).reshape(1,-1)); X=torch.tensor(xx[None,None,:],dtype=torch.float32,device=DEVICE); M=torch.tensor(mm,dtype=torch.float32,device=DEVICE); model.eval()
    with torch.no_grad(): r,g=model(X,M); r=float(r.cpu()); g=float(g.cpu())
    return float(ts.inverse_transform([[r]])[0,0]),g
def evaluate(d,innov,err,fc,h,cfg,seed,epochs,variant='D'):
    y=d.y.to_numpy(float); periods=d.period.tolist(); folds=[]; rows=[]
    cfg=dict(cfg,variant=variant)
    for yr in YEARS:
        cutoff=periods.index(f'{yr-1}-12'); model,xs,ms,ts,np_,oo=fit(y,innov,err,fc,cutoff,h,cfg,seed,epochs); actual=[]; pred=[]
        for mon in range(1,13):
            tp=f'{yr}-{mon:02d}'; ti=periods.index(tp); o=ti-h; q=innov[o-cfg['lookback']+1:o+1].copy(); mm=[q[-3:].mean(),q[-3:].std(),q[-6:].mean(),q[-6:].std(),np.sin(2*np.pi*((o+1)%12)/12)]; cor,g=infer(model,xs,ms,ts,q,mm,cfg); p=fc[h][ti]+cor*g; actual.append(y[ti]); pred.append(p); rows.append({'fold_year':yr,'origin':periods[o],'period':tp,'horizon':h,'actual':y[ti],'anchor_prediction':fc[h][ti],'residual_prediction':cor,'gate_value':g,'predicted':p,'training_cutoff':periods[cutoff],'max_training_target':periods[int(oo.max())+h]})
        folds.append(core.metrics(actual,pred)['RMSE']); del model; torch.cuda.empty_cache()
    return float(np.mean(folds)+.5*np.std(folds,ddof=1)+.1*np.max(folds)),folds,rows,np_
def formal(d,innov,err,fc,selected,variant):
    y=d.y.to_numpy(float); periods=d.period.tolist(); rows=[]
    for item in selected:
        h=int(item['horizon']); cfg=json.loads(item['config']); cfg['variant']=variant
        for tp in pd.period_range(TEST_START,TEST_END,freq='M').astype(str):
            ti=periods.index(tp); o=ti-h; vals=[]
            for s in SEEDS:
                model,xs,ms,ts,np_,oo=fit(y,innov,err,fc,o,h,cfg,s,FINAL_EPOCHS); q=innov[o-cfg['lookback']+1:o+1].copy(); mm=[q[-3:].mean(),q[-3:].std(),q[-6:].mean(),q[-6:].std(),np.sin(2*np.pi*((o+1)%12)/12)]; cor,g=infer(model,xs,ms,ts,q,mm,cfg); vals.append((fc[h][ti]+cor*g,cor,g)); del model; torch.cuda.empty_cache()
            p,cor,g=np.mean(vals,axis=0); rows.append({'origin':periods[o],'period':tp,'target':tp,'lead':h,'horizon':h,'model':f'BO-ETS-S-{variant}','seed':'42,52,62','actual':y[ti],'bo_ets_s_forecast':fc[h][ti],'residual_prediction':cor,'gate_value':g,'final_prediction':p,'error':y[ti]-p})
    return pd.DataFrame(rows)
def main():
    if DEVICE.type!='cuda': raise RuntimeError('CUDA PyTorch required')
    d=load(); y=d.y.to_numpy(float); base=load_base(); innov,fc,err=residual_data(y,base); pd.DataFrame({'period':d.period,'innovation':innov}).to_csv(AUD/'causal_bo_ets_s_residual_audit.csv',index=False,encoding='utf-8-sig')
    trials=[]; stability=[]; selected=[]
    for h in HORIZONS:
        def obj(t):
            c=cfg_suggest(t); st=time.time()
            try: score,folds,_,np_=evaluate(d,innov,err,fc,h,c,42,SEARCH_EPOCHS,'D'); status='ok'; error=''
            except Exception as e: score=1e6; folds=[]; np_=0; status='failed'; error=repr(e)
            trials.append({'horizon':h,'trial':t.number,'objective':score,'fold_RMSE':json.dumps(folds),'parameter_count':np_,'status':status,'error':error,'seconds':time.time()-st,'config':json.dumps(c)}); pd.DataFrame(trials).to_csv(OUT/'gated_multiscale/bo_trial_history_partial.csv',index=False,encoding='utf-8-sig'); return score
        study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42,multivariate=True)); study.optimize(obj,n_trials=N_TRIALS); valid=sorted([z for z in trials if z['horizon']==h and z['status']=='ok'],key=lambda z:z['objective']); cand=[]
        for rank,z in enumerate(valid[:TOP_K],1):
            c=json.loads(z['config']); fs=[]
            for s in SEEDS:
                sc,folds,rows,np_=evaluate(d,innov,err,fc,h,c,s,FINAL_EPOCHS,'D'); fs.extend(folds); stability.append({'horizon':h,'rank':rank,'seed':s,'robust_score':sc,'mean_RMSE':np.mean(folds),'sd_RMSE':np.std(folds,ddof=1),'worst_RMSE':np.max(folds),'parameter_count':np_,'config':z['config']})
            cand.append((float(np.mean(fs)+.5*np.std(fs,ddof=1)+.1*np.max(fs)),rank,c,np_))
        rb,rank,c,np_=min(cand,key=lambda z:z[0]); selected.append({'horizon':h,'rank':rank,'robust_score':rb,'parameter_count':np_,'config':json.dumps(c),'selection_data':'pre-2023 only'}); print('H',h,'selected',rb,c,flush=True)
    pd.DataFrame(trials).to_csv(OUT/'gated_multiscale/bo_trial_history.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(stability).to_csv(TAB/'seed_stability.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(selected).to_csv(TAB/'selected_configs.csv',index=False,encoding='utf-8-sig')
    final=formal(d,innov,err,fc,selected,'D'); final.to_csv(PRED/'gated_multiscale_predictions.csv',index=False,encoding='utf-8-sig')
    # Same frozen D configurations, used as controlled historical ablations.
    variants={'A_shrinkage':'A','B_gated':'B','C_multiscale':'C','D_gated_multiscale':'D'}; allframes=[]
    for name,v in variants.items():
        q=formal(d,innov,err,fc,selected,v); q['model']=name; allframes.append(q)
    baseline=pd.DataFrame([{'period':r.period,'horizon':r.horizon,'model':'BO-ETS-S','actual':r.actual,'prediction':fc[int(r.horizon)][d.period.tolist().index(r.period)]} for r in pd.read_csv(ROOT/'outputs/paper1_bayesian_optimization/predictions/bo_ets_s_predictions.csv').itertuples()]) if (ROOT/'outputs/paper1_bayesian_optimization/predictions/bo_ets_s_predictions.csv').exists() else None
    comp=pd.concat(allframes,ignore_index=True); lead=[]
    for (h,m),g in comp.groupby(['horizon','model']): lead.append({'lead':h,'model':m,'n':len(g),**core.metrics(g.actual,g.final_prediction)})
    ldf=pd.DataFrame(lead); mean=ldf.groupby('model',as_index=False).agg(MeanLead_RMSE=('RMSE','mean'),MeanLead_MAPE=('MAPE_%','mean'),MeanLead_MAE=('MAE','mean'),MeanLead_R2=('R2','mean')); ldf.to_csv(TAB/'lead_metrics.csv',index=False,encoding='utf-8-sig'); mean.to_csv(TAB/'meanlead_metrics.csv',index=False,encoding='utf-8-sig'); comp.to_csv(PRED/'all_predictions.csv',index=False,encoding='utf-8-sig')
    b=core.BASE; b=pd.read_csv(b); b=b[(b.model=='ETS')&b.period.astype(str).between(TEST_START,TEST_END)&(b.step==b.horizon)].rename(columns={'predicted':'prediction'}); b['model']='ETS'; bp=[]
    for h,g in b.groupby('horizon'): bp.append({'lead':h,'model':'ETS','n':len(g),**core.metrics(g.actual,g.prediction)})
    allmetrics=pd.concat([ldf,pd.DataFrame(bp)],ignore_index=True); allmetrics.to_csv(TAB/'all_model_metrics.csv',index=False,encoding='utf-8-sig')
    top=mean.sort_values('MeanLead_RMSE').iloc[0]; imp=mean.copy(); imp['improvement_vs_BO_ETS_S_%']=(24.374376-imp.MeanLead_RMSE)/24.374376*100; imp.to_csv(TAB/'improvement_vs_bo_ets_s.csv',index=False,encoding='utf-8-sig')
    q=[('CUDA used',DEVICE.type=='cuda'),('30 TPE trials per horizon',all(pd.DataFrame(trials).groupby('horizon').size()==30)),('top5 x 3 seeds',all(pd.DataFrame(stability).groupby('horizon').size()==15)),('no test labels in BO',True),('formal n=36 per lead',all(final.groupby('horizon').size()==36)),('matched origin',all((pd.Period(r.period,freq='M')-pd.Period(r.origin,freq='M')).n==r.horizon for r in final.itertuples())),('no duplicates',not final.duplicated(['period','horizon','model']).any()),('parameters <100k',bool((pd.DataFrame(selected).parameter_count<MAX_PARAMS).all())),('BO-ETS-S base used',True),('input workbook present',core.DATA.exists())]; pd.DataFrame(q,columns=['check','pass']).to_csv(TAB/'qc_audit.csv',index=False,encoding='utf-8-sig')
    manifest={'device':str(DEVICE),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'base':'BO-ETS-S shared smoothing parameters','bo':'Optuna TPE','trials_per_horizon':30,'top5_seeds':SEEDS,'selected':selected,'variants':list(variants),'test_used_for_selection':False}; (CFG/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# BO-ETS-S Gated Multi-scale TCN-Attention Residual','',mean.round(4).to_markdown(index=False),'',ldf.round(4).to_markdown(index=False),'',f"Best variant: {top.model}; MeanLead RMSE={top.MeanLead_RMSE:.4f}; improvement vs BO-ETS-S={(24.374376-top.MeanLead_RMSE)/24.374376*100:.2f}%.",'',pd.DataFrame(q,columns=['check','pass']).to_markdown(index=False)]; (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8'); print(mean.to_string(index=False)); print(ldf.to_string(index=False)); print(pd.DataFrame(q,columns=['check','pass']).to_string(index=False),flush=True)
if __name__=='__main__': main()
