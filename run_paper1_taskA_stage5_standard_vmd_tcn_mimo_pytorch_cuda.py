from __future__ import annotations
import json, platform, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from run_paper1_taskA_stage4_strict_ewt_tcn_mimo_pytorch_cuda import TCN, DEVICE, INTERNAL, HORIZONS, SEEDS, TEST_START, TEST_END, config_map, load, metrics, seed_all, fit_component
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs/paper1_taskA_stage5_standard_vmd_tcn_mimo_pytorch_cuda"; TAB=OUT/"tables"
for p in (OUT,TAB,OUT/"figures"): p.mkdir(parents=True,exist_ok=True)
KS=[3,4,5,6]; ALPHAS=[500,1000,2000,5000]; VMD_TOL=1e-7; VMD_MAX_ITER=500

def vmd(signal,K,alpha,tau=0,dc=False,init=1,tol=VMD_TOL):
    f=np.asarray(signal,float); n=len(f); ifftshift=np.fft.ifftshift; fftshift=np.fft.fftshift
    f=f-f.mean(); fs=f
    T=len(fs); t=np.arange(1,T+1); freqs=t/T-0.5
    f_hat=fftshift(np.fft.fft(fs)); f_hat_plus=f_hat.copy(); f_hat_plus[:T//2]=0
    N=VMD_MAX_ITER; u_hat=np.zeros((N,T,K),complex); omega=np.zeros((N,K)); lam=np.zeros((N,T),complex)
    if init==1: omega[0]=np.arange(K)*0.5/K
    elif init==2: omega[0]=np.sort(np.exp(np.log(1/T)+(np.log(0.5)-np.log(1/T))*np.random.rand(K)))
    else: omega[0]=0
    if dc: omega[0,0]=0
    u_diff=tol+1; n=0
    while u_diff>tol and n<N-1:
        u_hat[n+1,:,0]=(f_hat_plus-lam[n]/2)/(1+alpha*(freqs-omega[n,0])**2)
        if dc: omega[n+1,0]=0
        else:
            w=np.abs(u_hat[n+1,T//2:,0])**2; omega[n+1,0]=np.sum(freqs[T//2:]*w)/max(np.sum(w),1e-15)
        for k in range(1,K):
            residual=f_hat_plus-np.sum(u_hat[n+1,:,:k],axis=1)-lam[n]/2
            u_hat[n+1,:,k]=residual/(1+alpha*(freqs-omega[n,k])**2)
            w=np.abs(u_hat[n+1,T//2:,k])**2; omega[n+1,k]=np.sum(freqs[T//2:]*w)/max(np.sum(w),1e-15)
        lam[n+1]=lam[n]+tau*(np.sum(u_hat[n+1],axis=1)-f_hat_plus)
        n+=1; u_diff=np.finfo(float).eps
        for k in range(K): u_diff+=np.sum(np.abs(u_hat[n,:,k]-u_hat[n-1,:,k])**2)/max(np.sum(np.abs(u_hat[n-1,:,k])**2),1e-15)
    u_hat_final=u_hat[n]
    u=np.zeros((K,T))
    for k in range(K): u[k]=np.real(np.fft.ifft(ifftshift(u_hat_final[:,k])))
    # Add the removed DC level to the lowest-frequency mode and enforce exact reconstruction.
    u[0]+=np.mean(signal); u[-1]+=np.asarray(signal)-u.sum(axis=0)
    centers=np.asarray(omega[n],float); order=np.argsort(centers); u=u[order]; centers=centers[order]
    recon=u.sum(axis=0); err=float(np.linalg.norm(np.asarray(signal)-recon)/max(np.linalg.norm(signal),1e-12))
    return u,centers,n+1,err

def predict(y,oi,H,cfg,K,alpha,seed):
    hist=y[:oi+1]; modes,centers,nit,err=vmd(hist,K,alpha); ps=[]; params=0
    for mode in modes:
        p,np_=fit_component(mode,len(mode)-1,H,cfg,seed,epochs=100); ps.append(p); params+=np_
    return np.sum(ps,axis=0),centers,nit,err,modes,params

def tune(d,H,cfg):
    rows=[]
    for K in KS:
        for alpha in ALPHAS:
            fs=[]; failed=""
            for origin in INTERNAL:
                oi=int(d.index[d.period==origin][0]); fut=d.y.iloc[oi+1:oi+1+H].to_numpy(float)
                try: p,_,_,_,_,par=predict(d.y.to_numpy(float),oi,H,cfg,K,alpha,42); fs.append(metrics(fut,p))
                except Exception as e: failed=str(e); break
            if fs: rows.append({"horizon":H,"K":K,"alpha":alpha,"status":"ok","mean_RMSE":np.mean([x["RMSE"] for x in fs]),"sd_RMSE":np.std([x["RMSE"] for x in fs],ddof=1),"mean_MAPE_%":np.mean([x["MAPE_%"] for x in fs]),"n_folds":len(fs),"component_parameter_count":par})
            else: rows.append({"horizon":H,"K":K,"alpha":alpha,"status":"failed","error":failed})
    cv=pd.DataFrame(rows); ok=cv[cv.status=="ok"].sort_values(["mean_RMSE","mean_MAPE_%"]); return ok.iloc[0].to_dict(),cv

def main():
    d=load(); periods=d.period.tolist(); first=periods.index(TEST_START); last=periods.index(TEST_END); cfgs=config_map(); best=[]; cvs=[]
    for H in HORIZONS:
        b,cv=tune(d,H,cfgs[H]); best.append(b);cvs.append(cv)
    sel=pd.DataFrame(best); sel.to_csv(TAB/"taskA_stage5_selected_vmd_parameters.csv",index=False,encoding="utf-8-sig");pd.concat(cvs,ignore_index=True).to_csv(TAB/"taskA_stage5_internal_cv_records.csv",index=False,encoding="utf-8-sig")
    pmap={int(r.horizon):(int(r.K),float(r.alpha)) for _,r in sel.iterrows()}; rec=[]; audit=[]; failures=[]
    for H in HORIZONS:
        cfg=cfgs[H];K,alpha=pmap[H]
        for oi in range(max(cfg["lookback"]-1,first-H),last):
            origin=periods[oi]; fut=d.iloc[oi+1:oi+1+H]
            if len(fut)<H: continue
            for seed in SEEDS:
                try:
                    p,centers,nit,err,modes,par=predict(d.y.to_numpy(float),oi,H,cfg,K,alpha,seed);status,e="ok",""; energy=(np.sum(modes**2,axis=1)/np.sum(modes**2)).tolist()
                    audit.append({"origin":origin,"horizon":H,"seed":seed,"K":K,"alpha":alpha,"reconstruction_error":err,"converged":nit<VMD_MAX_ITER,"n_iterations":nit,"center_frequencies":json.dumps(centers.tolist()),"mode_energy_ratios":json.dumps(energy),"mode_variances":json.dumps(np.var(modes,axis=1).tolist()),"mode_order_after_sort":"ascending_center_frequency","history_end":origin})
                except Exception as ex: p=np.repeat(d.y.iloc[oi],H);par=0;status,e="failed",str(ex);failures.append({"origin":origin,"horizon":H,"seed":seed,"K":K,"alpha":alpha,"error":e})
                for j,(_,r) in enumerate(fut.iterrows(),1):
                    if TEST_START<=r.period<=TEST_END: rec.append({"origin":origin,"period":r.period,"horizon":H,"step":j,"model":"StandardVMD_TCN_MIMO_PyTorch_CUDA","seed":seed,"actual":float(r.y),"predicted":float(p[j-1]),"status":status,"error":e,"K":K,"alpha":alpha,"parameter_count":par})
    sd=pd.DataFrame(rec);sd.to_csv(TAB/"taskA_stage5_seed_predictions.csv",index=False,encoding="utf-8-sig");pd.DataFrame(audit).to_csv(TAB/"taskA_stage5_decomposition_audit.csv",index=False,encoding="utf-8-sig");pd.DataFrame(failures).to_csv(TAB/"taskA_stage5_failure_audit.csv",index=False,encoding="utf-8-sig")
    final=sd[sd.status=="ok"].groupby(["origin","period","horizon","step","model","actual"],as_index=False).predicted.mean().rename(columns={"predicted":"predicted_final"});final.to_csv(TAB/"taskA_stage5_predictions.csv",index=False,encoding="utf-8-sig")
    leads=[];blocks=[];steps=[]
    for H in HORIZONS:
        for m,g in final[final.horizon==H].groupby("model"):
            z=g[g.step==H];leads.append({"lead":H,"model":m,"n":len(z),**metrics(z.actual,z.predicted_final)});blocks.append({"block_horizon":H,"model":m,"n":len(g),**metrics(g.actual,g.predicted_final)})
            for st,w in g.groupby("step"): steps.append({"block_horizon":H,"step":st,"model":m,"n":len(w),**metrics(w.actual,w.predicted_final)})
    lead=pd.DataFrame(leads);block=pd.DataFrame(blocks);step=pd.DataFrame(steps);lead.to_csv(TAB/"taskA_stage5_lead_specific_metrics.csv",index=False,encoding="utf-8-sig");block.to_csv(TAB/"taskA_stage5_block_metrics_raw.csv",index=False,encoding="utf-8-sig");step.to_csv(TAB/"taskA_stage5_stepwise_metrics.csv",index=False,encoding="utf-8-sig")
    mean=lead.groupby("model").agg(MeanLead_MAE=("MAE","mean"),MeanLead_RMSE=("RMSE","mean"),MeanLead_MAPE=("MAPE_%","mean"),MeanLead_sMAPE=("sMAPE_%","mean"),MeanLead_R2=("R2","mean")).reset_index();mean.to_csv(TAB/"taskA_stage5_mean_across_leads.csv",index=False,encoding="utf-8-sig")
    ets=pd.read_csv(ROOT/"outputs/paper1_taskA_stage2_traditional_models/tables/taskA_stage2_lead_specific_metrics.csv");ets=ets[ets.model=="ETS"].set_index("lead");imp=lead.copy();imp["RMSE_improvement_vs_ETS_%"]=imp.apply(lambda r:(ets.loc[r.lead,"RMSE"]-r.RMSE)/ets.loc[r.lead,"RMSE"]*100,axis=1);imp["MAPE_improvement_vs_ETS_pp"]=imp.apply(lambda r:ets.loc[r.lead,"MAPE_%"]-r["MAPE_%"],axis=1);imp.to_csv(TAB/"taskA_stage5_improvement_vs_ETS.csv",index=False,encoding="utf-8-sig")
    key=final.groupby(["origin","period","horizon","step","model"]).size().reset_index(name="n_rows");key.to_csv(TAB/"taskA_stage5_uniqueness_audit.csv",index=False,encoding="utf-8-sig")
    man={"version":"Task A Stage 5 Standard VMD-TCN-MIMO PyTorch CUDA","device":str(DEVICE),"pytorch":torch.__version__,"cuda":torch.version.cuda,"target":"shanghai_teu_10k","test":f"{TEST_START} to {TEST_END}","horizons":HORIZONS,"seeds":SEEDS,"internal_origins":INTERNAL,"K_candidates":KS,"alpha_candidates":ALPHAS,"selected":best,"architecture_source":"Stage 3 frozen horizon-specific TCN configurations","vmd_tol":VMD_TOL,"vmd_max_iter":VMD_MAX_ITER,"causal_decomposition":True,"mode_order":"ascending center frequency","failure_count":len(failures),"duplicate_keys":int((key.n_rows>1).sum()),"python":platform.python_version()};(OUT/"taskA_stage5_manifest.json").write_text(json.dumps(man,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    report=["# Task A Stage 5: Standard VMD-TCN-MIMO (PyTorch CUDA)","",f"Device: `{DEVICE}`; PyTorch: `{torch.__version__}`; CUDA: `{torch.version.cuda}`.","","VMD was recomputed at every forecast origin using history through that origin only. Modes were sorted by ascending center frequency. Stage 3 horizon-specific TCN architectures were frozen; VMD K and alpha were selected only on internal origins through 2021-12.","","## Selected VMD parameters","",sel.to_markdown(index=False),"","## Lead-specific results","",lead.round(4).to_markdown(index=False),"","## Mean across leads","",mean.round(4).to_markdown(index=False),"","## Audits","",f"Decomposition records: {len(audit)}; failed forecasts: {len(failures)}; duplicate prediction keys: {int((key.n_rows>1).sum())}; lead counts: {lead[['lead','n']].to_dict('records') if len(lead) else []}.","","## Files","",f"- Report: `{OUT/'taskA_stage5_standard_vmd_tcn_mimo_report.md'}`",f"- Lead metrics: `{TAB/'taskA_stage5_lead_specific_metrics.csv'}`",f"- Predictions: `{TAB/'taskA_stage5_predictions.csv'}`",f"- VMD audit: `{TAB/'taskA_stage5_decomposition_audit.csv'}`",f"- Manifest: `{OUT/'taskA_stage5_manifest.json'}`"]
    (OUT/"taskA_stage5_standard_vmd_tcn_mimo_report.md").write_text("\n".join(report),encoding="utf-8");print(lead.to_string(index=False));print("MEAN\n",mean.to_string(index=False),flush=True)

if __name__=="__main__": main()
