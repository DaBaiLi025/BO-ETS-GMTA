from __future__ import annotations
import json, platform, random, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from torch import nn
warnings.filterwarnings("ignore")

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/clean_model_dataset_v5_shanghai_augmented.xlsx"
S3=ROOT/"outputs/paper1_taskA_stage3_raw_tcn_mimo_pytorch_cuda/tables/taskA_stage3_selected_hyperparameters.csv"
ETS=ROOT/"outputs/paper1_taskA_stage2_traditional_models/tables/taskA_stage2_lead_specific_metrics.csv"
OUT=ROOT/"outputs/paper1_taskA_stage4_strict_ewt_tcn_mimo_pytorch_cuda"; TAB=OUT/"tables"; FIG=OUT/"figures"
for p in (OUT,TAB,FIG): p.mkdir(parents=True,exist_ok=True)
TEST_START,TEST_END="2023-01","2025-12"; HORIZONS=[1,3,6,12]; SEEDS=[42,52,62]; INTERNAL=["2018-12","2019-12","2020-12","2021-12"]
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def load():
    d=pd.read_excel(DATA,sheet_name="Clean_Main_2012_2025"); d.period=d.period.astype(str); d=d.sort_values("period").drop_duplicates("period")
    d["y"]=pd.to_numeric(d.shanghai_teu_10k,errors="coerce"); return d[d.y.notna()].reset_index(drop=True)

def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); den=np.abs(y)+np.abs(p)
    return {"MAE":float(np.mean(np.abs(y-p))),"RMSE":float(np.sqrt(np.mean((y-p)**2))),"MAPE_%":float(np.mean(np.abs((y-p)/y))*100),"sMAPE_%":float(np.mean(np.where(den==0,0,2*np.abs(y-p)/den))*100),"R2":float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))}

def ewt(x,k):
    """Causal empirical frequency-band decomposition; x contains history through one origin only."""
    x=np.asarray(x,float); n=len(x); spec=np.abs(np.fft.rfft(x-x.mean())); smooth=gaussian_filter1d(spec,1.2); freq=np.fft.rfftfreq(n)
    if len(smooth)>4:
        peaks,_=find_peaks(smooth[1:],distance=max(2,len(smooth)//(k*3)),prominence=max(np.max(smooth)*.01,1e-9)); peaks=peaks+1
    else: peaks=np.array([],int)
    candidates=sorted(peaks.tolist(),key=lambda i: smooth[i],reverse=True)[:max(0,k-1)]
    if len(candidates)<k-1:
        candidates=list(np.linspace(1,max(1,len(freq)-2),k-1,dtype=int))
    centers=sorted(set(candidates))
    while len(centers)<k-1:
        centers.append(min(len(freq)-2,1+len(centers)))
        centers=sorted(set(centers))
    centers=centers[:k-1]; bounds=[0]+[float((freq[a]+freq[b])/2) for a,b in zip(centers[:-1],centers[1:])]+[float((freq[centers[-1]]+freq[-1])/2),.5]
    if k==2: bounds=[0,float((freq[centers[0]]+freq[-1])/2),.5]
    bands=[]
    for i in range(k):
        lo,hi=bounds[i],bounds[i+1]; mask=(freq>=lo)&((freq<hi) if i<k-1 else (freq<=hi)); spectrum=np.fft.rfft(x-x.mean()); z=np.zeros_like(spectrum); z[mask]=spectrum[mask]
        comp=np.fft.irfft(z,n=n)+x.mean()/k; bands.append(comp)
    # Correct numerical/mean allocation so the decomposition is exactly additive.
    bands=np.asarray(bands); bands[-1]+=x-bands.sum(axis=0)
    return bands,bounds

class CausalConv(nn.Module):
    def __init__(self,ci,co,ks,d): super().__init__(); self.left=(ks-1)*d; self.conv=nn.Conv1d(ci,co,ks,dilation=d)
    def forward(self,x): return self.conv(nn.functional.pad(x,(self.left,0)))

class TCN(nn.Module):
    def __init__(self,cfg,h):
        super().__init__(); self.blocks=nn.ModuleList(); self.proj=nn.ModuleList(); ci=1
        for bi in range(cfg["residual_blocks"]):
            b=nn.ModuleList()
            for di in cfg["dilations"]:
                b.extend([CausalConv(ci,cfg["filters"],cfg["kernel_size"],di),nn.ReLU()]);
                if cfg["dropout"]: b.append(nn.Dropout(cfg["dropout"]))
                ci=cfg["filters"]
            self.blocks.append(b); self.proj.append(nn.Conv1d(1 if bi==0 else cfg["filters"],cfg["filters"],1))
        self.head=nn.Linear(cfg["lookback"]*cfg["filters"],h)
    def forward(self,x):
        for b,pr in zip(self.blocks,self.proj):
            r=x
            for layer in b: x=layer(x)
            if r.shape[1]!=x.shape[1]: r=pr(r)
            x=torch.relu(x+r)
        return self.head(x.flatten(1))

def samples(v,L,H,oi):
    X=[];Y=[]
    for end in range(L-1,oi-H+1): X.append(v[end-L+1:end+1]);Y.append(v[end+1:end+H+1])
    return np.asarray(X),np.asarray(Y)

def fit_component(v,oi,H,cfg,seed,epochs=100):
    seed_all(seed); sc=StandardScaler().fit(np.asarray(v[:oi+1]).reshape(-1,1)); z=sc.transform(np.asarray(v[:oi+1]).reshape(-1,1)).ravel(); X,Y=samples(z,cfg["lookback"],H,oi)
    if len(X)<12: raise ValueError("fewer than 12 causal component samples")
    cut=max(1,int(.8*len(X))); xt=torch.tensor(X[:cut,None,:],dtype=torch.float32,device=DEVICE); yt=torch.tensor(Y[:cut],dtype=torch.float32,device=DEVICE)
    xv=torch.tensor(X[cut:,None,:],dtype=torch.float32,device=DEVICE) if cut<len(X) else None; yv=torch.tensor(Y[cut:],dtype=torch.float32,device=DEVICE) if cut<len(X) else None
    m=TCN(cfg,H).to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=.001); loss=nn.MSELoss(); best=None; bestv=float("inf"); wait=0
    for _ in range(epochs):
        m.train(); opt.zero_grad(set_to_none=True); loss( m(xt),yt).backward(); opt.step(); m.eval()
        with torch.no_grad(): vl=float(loss(m(xv),yv)) if xv is not None else float(loss(m(xt),yt))
        if vl<bestv-1e-5: bestv=vl;wait=0;best={k:v.detach().clone() for k,v in m.state_dict().items()}
        else: wait+=1
        if wait>=15: break
    if best: m.load_state_dict(best)
    last=torch.tensor(z[-cfg["lookback"]:][None,None,:],dtype=torch.float32,device=DEVICE)
    with torch.no_grad(): p=m.eval()(last).cpu().numpy().ravel()
    return sc.inverse_transform(p[:,None]).ravel(),sum(q.numel() for q in m.parameters())

def config_map():
    q=pd.read_csv(S3); return {int(r.horizon):json.loads(r.config) for _,r in q.iterrows()}

def decompose_history(y,oi,k): return ewt(y[:oi+1],k)

def predict_ewt(y,oi,H,cfg,k,seed):
    comps,bounds=decompose_history(y,oi,k); preds=[]; params=0
    for c in comps:
        p,n=fit_component(c,oi,H,cfg,seed,epochs=100); preds.append(p);params+=n
    return np.sum(preds,axis=0),bounds,comps,params

def tune(d,H,cfg):
    rows=[]
    for k in (3,4,5):
        fs=[]
        for origin in INTERNAL:
            oi=int(d.index[d.period==origin][0]); fut=d.y.iloc[oi+1:oi+1+H].to_numpy(float)
            try: p,_,_,par=predict_ewt(d.y.to_numpy(float),oi,H,cfg,k,42);fs.append({"origin":origin,**metrics(fut,p)})
            except Exception as e: fs=[]; rows.append({"horizon":H,"n_components":k,"status":"failed","error":str(e)});break
        if fs: rows.append({"horizon":H,"n_components":k,"status":"ok","mean_RMSE":np.mean([z["RMSE"] for z in fs]),"sd_RMSE":np.std([z["RMSE"] for z in fs],ddof=1),"mean_MAPE_%":np.mean([z["MAPE_%"] for z in fs]),"n_folds":len(fs),"parameter_count":par})
    cv=pd.DataFrame(rows); ok=cv[cv.status=="ok"].sort_values(["mean_RMSE","mean_MAPE_%"]); return int(ok.iloc[0].n_components),cv

def main():
    d=load(); periods=d.period.tolist(); first=periods.index(TEST_START); last=periods.index(TEST_END); cfgs=config_map(); selected=[]; cvall=[]
    for H in HORIZONS:
        k,cv=tune(d,H,cfgs[H]); selected.append({"horizon":H,"n_components":k});cvall.append(cv)
    sel=pd.DataFrame(selected); sel.to_csv(TAB/"taskA_stage4_selected_components.csv",index=False,encoding="utf-8-sig");pd.concat(cvall,ignore_index=True).to_csv(TAB/"taskA_stage4_internal_cv_records.csv",index=False,encoding="utf-8-sig")
    cmap={int(r.horizon):int(r.n_components) for _,r in sel.iterrows()}; records=[]; decomp=[]; audits=[]
    for H in HORIZONS:
        cfg=cfgs[H]; k=cmap[H]
        for oi in range(max(cfg["lookback"]-1,first-H),last):
            origin=periods[oi]; fut=d.iloc[oi+1:oi+1+H]
            if len(fut)<H: continue
            for seed in SEEDS:
                try:
                    p,bounds,comps,par=predict_ewt(d.y.to_numpy(float),oi,H,cfg,k,seed); status,err="ok",""
                    e=np.sum(comps**2,axis=1); er=(e/e.sum()).tolist(); decomp.append({"origin":origin,"horizon":H,"seed":seed,"n_components":k,"frequency_boundaries":json.dumps(bounds),"component_energy_ratio":json.dumps(er),"history_end":origin})
                except Exception as e: p=np.repeat(d.y.iloc[oi],H);par=0;status,err="failed",str(e); audits.append({"origin":origin,"horizon":H,"seed":seed,"error":err})
                for j,(_,r) in enumerate(fut.iterrows(),1):
                    if TEST_START<=r.period<=TEST_END: records.append({"origin":origin,"period":r.period,"horizon":H,"step":j,"model":"EWT_TCN_MIMO_PyTorch_CUDA","seed":seed,"actual":float(r.y),"predicted":float(p[j-1]),"status":status,"error":err,"n_components":k,"parameter_count":par})
    sd=pd.DataFrame(records);sd.to_csv(TAB/"taskA_stage4_seed_predictions.csv",index=False,encoding="utf-8-sig");pd.DataFrame(decomp).to_csv(TAB/"taskA_stage4_decomposition_audit.csv",index=False,encoding="utf-8-sig");pd.DataFrame(audits).to_csv(TAB/"taskA_stage4_failure_audit.csv",index=False,encoding="utf-8-sig")
    final=sd[sd.status=="ok"].groupby(["origin","period","horizon","step","model","actual"],as_index=False).predicted.mean().rename(columns={"predicted":"predicted_final"});final.to_csv(TAB/"taskA_stage4_predictions.csv",index=False,encoding="utf-8-sig")
    leads=[];blocks=[];steps=[]
    for H in HORIZONS:
        g=final[final.horizon==H]
        for m,z in g.groupby("model"):
            zz=z[z.step==H];leads.append({"lead":H,"model":m,"n":len(zz),**metrics(zz.actual,zz.predicted_final)});blocks.append({"block_horizon":H,"model":m,"n":len(z),**metrics(z.actual,z.predicted_final)})
            for st,w in z.groupby("step"):steps.append({"block_horizon":H,"step":st,"model":m,"n":len(w),**metrics(w.actual,w.predicted_final)})
    lead=pd.DataFrame(leads);block=pd.DataFrame(blocks);step=pd.DataFrame(steps);lead.to_csv(TAB/"taskA_stage4_lead_specific_metrics.csv",index=False,encoding="utf-8-sig");block.to_csv(TAB/"taskA_stage4_block_metrics_raw.csv",index=False,encoding="utf-8-sig");step.to_csv(TAB/"taskA_stage4_stepwise_metrics.csv",index=False,encoding="utf-8-sig")
    mean=lead.groupby("model").agg(MeanLead_MAE=("MAE","mean"),MeanLead_RMSE=("RMSE","mean"),MeanLead_MAPE=("MAPE_%","mean"),MeanLead_sMAPE=("sMAPE_%","mean"),MeanLead_R2=("R2","mean")).reset_index();mean.to_csv(TAB/"taskA_stage4_mean_across_leads.csv",index=False,encoding="utf-8-sig")
    et=pd.read_csv(ETS);et=et[et.model=="ETS"].set_index("lead");imp=lead.copy();imp["RMSE_improvement_vs_ETS_%"]=imp.apply(lambda r:(et.loc[r.lead,"RMSE"]-r.RMSE)/et.loc[r.lead,"RMSE"]*100,axis=1);imp["MAPE_improvement_vs_ETS_pp"]=imp.apply(lambda r:et.loc[r.lead,"MAPE_%"]-r["MAPE_%"],axis=1);imp.to_csv(TAB/"taskA_stage4_improvement_vs_ETS.csv",index=False,encoding="utf-8-sig")
    key=final.groupby(["origin","period","horizon","step","model"]).size().reset_index(name="n_rows");key.to_csv(TAB/"taskA_stage4_uniqueness_audit.csv",index=False,encoding="utf-8-sig")
    for col,name,color in [("RMSE","RMSE","#4E79A7"),("MAPE_%","MAPE","#F28E2B")]:
        fig,ax=plt.subplots(figsize=(7.5,4.3));q=lead.sort_values("lead");ax.plot(q.lead,q[col],marker="o",lw=1.6,color=color,label="Strict EWT-TCN-MIMO");ax.plot(HORIZONS,[float(et.loc[h,col]) for h in HORIZONS],marker="o",lw=1.6,color="#59A14F",label="ETS");ax.set_xlabel("Forecast lead (months)");ax.set_ylabel(col);ax.set_title(f"Task A Stage 4 lead-specific {name}",loc="left",fontweight="bold");ax.legend();fig.tight_layout();fig.savefig(FIG/f"fig_stage4_lead_{name.lower()}.png",dpi=400);fig.savefig(FIG/f"fig_stage4_lead_{name.lower()}.svg");plt.close(fig)
    man={"version":"Task A Stage 4 Strict EWT-TCN-MIMO PyTorch CUDA","device":str(DEVICE),"pytorch":torch.__version__,"cuda":torch.version.cuda,"target":"shanghai_teu_10k","test":f"{TEST_START} to {TEST_END}","horizons":HORIZONS,"seeds":SEEDS,"internal_origins":INTERNAL,"component_candidates":[3,4,5],"selected_components":selected,"architecture_source":"Stage 3 frozen horizon-specific TCN configurations","causal_decomposition":True,"external_variables":False,"stage3_not_overwritten":True,"failure_count":len(audits),"python":platform.python_version()};(OUT/"taskA_stage4_manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    report=["# Task A Stage 4: Strict EWT-TCN-MIMO (PyTorch CUDA)","",f"Device: `{DEVICE}`; PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`.","","Each forecast origin was decomposed using only historical Shanghai TEU through that origin. The Stage 3 horizon-specific TCN architecture was frozen; only component count (3/4/5) was selected using internal origins through 2021-12. Component forecasts were summed and seed-averaged.","","## Selected component counts","",sel.to_markdown(index=False),"","## Lead-specific results","",lead.round(4).to_markdown(index=False),"","## Mean across leads","",mean.round(4).to_markdown(index=False),"","## Audits","",f"Decomposition records: {len(decomp)}; failed forecasts: {len(audits)}; duplicate prediction keys: {int((key.n_rows>1).sum())}; expected lead n=36: {lead[['lead','n']].to_dict('records') if len(lead) else []}.","","## Files","",f"- Report: `{OUT/'taskA_stage4_strict_ewt_tcn_mimo_report.md'}`",f"- Lead metrics: `{TAB/'taskA_stage4_lead_specific_metrics.csv'}`",f"- Predictions: `{TAB/'taskA_stage4_predictions.csv'}`",f"- Decomposition audit: `{TAB/'taskA_stage4_decomposition_audit.csv'}`",f"- Manifest: `{OUT/'taskA_stage4_manifest.json'}`"]
    (OUT/"taskA_stage4_strict_ewt_tcn_mimo_report.md").write_text("\n".join(report),encoding="utf-8");print(lead.to_string(index=False));print("MEAN\n",mean.to_string(index=False),flush=True)

if __name__=="__main__": main()
