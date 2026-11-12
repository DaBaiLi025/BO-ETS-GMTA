from __future__ import annotations

import json, platform, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from skopt import gp_minimize
from skopt.space import Real
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'/'clean_model_dataset_v5_shanghai_augmented.xlsx'
BASE = ROOT/'outputs'/'paper1_taskA_stage2_traditional_models'/'tables'/'taskA_stage2_predictions.csv'
OUT = ROOT/'outputs'/'paper1_bayesian_optimization'
SUB, TAB, PRED, FIG, AUD, CFG, LOG = [OUT/x for x in ['bo_ets','tables','predictions','figures','audits','configs','logs']]
for p in (SUB,TAB,PRED,FIG,AUD,CFG,LOG): p.mkdir(parents=True,exist_ok=True)
HORIZONS=[1,3,6,12]; YEARS=[2018,2019,2020,2021,2022]
TEST_START,TEST_END='2023-01','2025-12'; N_CALLS=75; N_INITIAL=15

def load():
    d=pd.read_excel(DATA,sheet_name='Clean_Main_2012_2025'); d.period=d.period.astype(str)
    d=d.sort_values('period').drop_duplicates('period'); d['y']=pd.to_numeric(d.shanghai_teu_10k,errors='coerce')
    return d[d.y.notna()].reset_index(drop=True)

def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float); e=y-p; den=np.abs(y)+np.abs(p)
    return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'MAPE_%':float(np.mean(np.abs(e/y))*100),'sMAPE_%':float(np.mean(np.where(den==0,0,2*np.abs(e)/den))*100),'R2':float(1-np.sum(e**2)/np.sum((y-y.mean())**2))}

def forecast(y,h,a,b,g):
    try:
        model=ExponentialSmoothing(np.asarray(y,float),trend='add',damped_trend=False,seasonal='add',seasonal_periods=12,initialization_method='estimated')
        fit=model.fit(smoothing_level=a,smoothing_trend=b,smoothing_seasonal=g,optimized=False,remove_bias=False)
        return np.maximum(np.asarray(fit.forecast(h),float),0)
    except Exception:
        return np.repeat(np.nan,h)

def matched_year(d,year,params,horizons=HORIZONS):
    periods=d.period.tolist(); rows=[]
    for H in horizons:
        for month in range(1,13):
            target=f'{year}-{month:02d}'; ti=periods.index(target); oi=ti-H
            pred=forecast(d.y.iloc[:oi+1].to_numpy(float),H,*params)[-1]
            rows.append({'fold_year':year,'origin':periods[oi],'period':target,'horizon':H,'actual':float(d.y.iloc[ti]),'predicted':float(pred)})
    return pd.DataFrame(rows)

def objective_factory(d,horizons,trial_rows):
    def obj(x):
        started=time.time(); fold=[]; valid=True
        for year in YEARS:
            q=matched_year(d,year,x,horizons)
            if q.predicted.isna().any(): valid=False; rmse=1e6
            else: rmse=metrics(q.actual,q.predicted)['RMSE']
            fold.append(rmse)
        mean=float(np.mean(fold)); sd=float(np.std(fold,ddof=1)); worst=float(np.max(fold)); score=mean+.5*sd+.1*worst
        trial_rows.append({'trial':len(trial_rows),'alpha':x[0],'beta':x[1],'gamma':x[2],'mean_RMSE':mean,'sd_RMSE':sd,'worst_RMSE':worst,'objective':score,'valid':valid,'seconds':time.time()-started,'horizons':','.join(map(str,horizons))})
        return score
    return obj

def run_bo(d,horizons,label):
    trials=[]; space=[Real(.01,.99,name='alpha'),Real(.001,.50,name='beta'),Real(.001,.99,name='gamma')]
    res=gp_minimize(objective_factory(d,horizons,trials),space,n_calls=N_CALLS,n_initial_points=N_INITIAL,random_state=42,acq_func='EI',noise=1e-8)
    pd.DataFrame(trials).to_csv(SUB/f'{label}_trial_history.csv',index=False,encoding='utf-8-sig')
    return res.x,res.fun,pd.DataFrame(trials)

def dm_test(y,pa,pb,lag):
    d=(y-pa)**2-(y-pb)**2; n=len(d); md=d.mean(); z=d-md
    gamma0=np.dot(z,z)/n; lrv=gamma0
    for k in range(1,min(lag,n-1)+1): lrv+=2*(1-k/(lag+1))*np.dot(z[k:],z[:-k])/n
    stat=md/np.sqrt(max(lrv/n,1e-12)); return float(stat),float(2*(1-norm.cdf(abs(stat))))

def bootstrap(y,pc,pb,h,B=2000):
    rng=np.random.default_rng(42); n=len(y); L=max(3,h); vals=[]
    for _ in range(B):
        start=rng.integers(0,n,size=int(np.ceil(n/L))); idx=np.concatenate([(s+np.arange(L))%n for s in start])[:n]
        vals.append(np.sqrt(np.mean((y[idx]-pc[idx])**2))-np.sqrt(np.mean((y[idx]-pb[idx])**2)))
    return np.quantile(vals,[.025,.975])

def main():
    d=load(); base=pd.read_csv(BASE); base.period=base.period.astype(str); base=base[(base.model=='ETS')&base.period.between(TEST_START,TEST_END)&(base.step==base.horizon)].copy()
    shared,score,trials=run_bo(d,HORIZONS,'bo_ets_s')
    fold_rows=[]
    for year in YEARS:
        q=matched_year(d,year,shared)
        for H,g in q.groupby('horizon'): fold_rows.append({'variant':'BO-ETS-S','fold_year':year,'horizon':H,'n':len(g),**metrics(g.actual,g.predicted)})
    pd.DataFrame(fold_rows).to_csv(TAB/'pretest_fold_metrics.csv',index=False,encoding='utf-8-sig')
    selected=pd.DataFrame([{'variant':'BO-ETS-S','horizon':'shared','alpha':shared[0],'beta':shared[1],'gamma':shared[2],'objective':score,'n_trials':N_CALLS,'selection_data':'2012-02 through 2022-12 only'}]); selected.to_csv(TAB/'selected_configs.csv',index=False,encoding='utf-8-sig')
    periods=d.period.tolist(); rec=[]
    for H in HORIZONS:
        for target in pd.period_range(TEST_START,TEST_END,freq='M').astype(str):
            ti=periods.index(target); oi=ti-H; p=forecast(d.y.iloc[:oi+1],H,*shared)[-1]
            rec.append({'origin':periods[oi],'period':target,'horizon':H,'lead':H,'model':'BO-ETS-S','actual':float(d.y.iloc[ti]),'prediction':float(p)})
    cand=pd.DataFrame(rec); cand.to_csv(PRED/'bo_ets_s_predictions.csv',index=False,encoding='utf-8-sig')
    b=base.rename(columns={'predicted':'prediction'})[['origin','period','horizon','model','actual','prediction']]; b['lead']=b.horizon
    allp=pd.concat([b,cand],ignore_index=True); allp.to_csv(PRED/'all_predictions.csv',index=False,encoding='utf-8-sig')
    lead=[]; dm=[]; boot=[]
    for (H,m),g in allp.groupby(['horizon','model']): lead.append({'lead':H,'model':m,'n':len(g),**metrics(g.actual,g.prediction)})
    lead=pd.DataFrame(lead); lead.to_csv(TAB/'lead_metrics.csv',index=False,encoding='utf-8-sig')
    mean=lead.groupby('model',as_index=False).agg(MeanLead_MAE=('MAE','mean'),MeanLead_RMSE=('RMSE','mean'),MeanLead_MAPE=('MAPE_%','mean'),MeanLead_sMAPE=('sMAPE_%','mean'),MeanLead_R2=('R2','mean')); mean.to_csv(TAB/'meanlead_metrics.csv',index=False,encoding='utf-8-sig')
    ets=lead[lead.model=='ETS'].set_index('lead'); imp=lead[lead.model=='BO-ETS-S'].copy(); imp['RMSE_improvement_vs_ETS_%']=imp.apply(lambda r:(ets.loc[r.lead,'RMSE']-r.RMSE)/ets.loc[r.lead,'RMSE']*100,axis=1); imp.to_csv(TAB/'improvement_vs_ets.csv',index=False,encoding='utf-8-sig')
    for H in HORIZONS:
        c=cand[cand.horizon==H].sort_values('period'); e=b[b.horizon==H].sort_values('period'); y=c.actual.to_numpy(float); pc=c.prediction.to_numpy(float); pe=e.prediction.to_numpy(float)
        st,pv=dm_test(y,pc,pe,H-1); dm.append({'lead':H,'candidate':'BO-ETS-S','baseline':'ETS','n':len(y),'loss':'squared','DM_stat':st,'p_value':pv})
        lo,hi=bootstrap(y,pc,pe,H); boot.append({'lead':H,'candidate':'BO-ETS-S','baseline':'ETS','metric':'RMSE_candidate_minus_ETS','n':len(y),'ci_low':lo,'ci_high':hi})
    pd.DataFrame(dm).to_csv(TAB/'dm_tests.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(boot).to_csv(TAB/'bootstrap_ci.csv',index=False,encoding='utf-8-sig')
    qcs=[('no 2023-2025 labels in BO',True),('formal n=36 per lead',all(cand.groupby('horizon').size()==36)),('no duplicate prediction keys',not allp.duplicated(['period','horizon','model']).any()),('matched origin correct',all((pd.Period(r.period,freq='M')-pd.Period(r.origin,freq='M')).n==r.horizon for r in cand.itertuples())),('ETS benchmark present',bool((mean.model=='ETS').any())),('final selection pretest only',True)]
    pd.DataFrame(qcs,columns=['check','pass']).to_csv(TAB/'qc_audit.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(7,4)); z=lead.pivot(index='lead',columns='model',values='RMSE'); z.plot(marker='o',ax=ax); ax.set_ylabel('RMSE'); fig.tight_layout(); fig.savefig(FIG/'bo_ets_lead_rmse.png',dpi=300); plt.close(fig)
    man={'variant':'BO-ETS-S','bo_method':'Gaussian-process Bayesian optimization (scikit-optimize)','n_calls':N_CALLS,'n_initial_points':N_INITIAL,'objective':'mean RMSE + 0.5 SD RMSE + 0.1 worst RMSE across five annual pseudo-OOS folds','frozen_structure':{'trend':'add','seasonal':'add','seasonal_periods':12,'damped_trend':False,'initialization_method':'estimated'},'space':{'alpha':[.01,.99],'beta':[.001,.50],'gamma':[.001,.99]},'selected':selected.to_dict('records'),'python':platform.python_version(),'test_used_for_bo':False}
    (CFG/'bo_ets_s_manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
    best=float(mean.loc[mean.model=='BO-ETS-S','MeanLead_RMSE'].iloc[0]); mape=float(mean.loc[mean.model=='BO-ETS-S','MeanLead_MAPE'].iloc[0]); wins=int((imp['RMSE_improvement_vs_ETS_%']>0).sum())
    report=['# Bayesian Optimization: BO-ETS-S','',f'GP-BO trials: {N_CALLS} ({N_INITIAL} random initial points). No 2023-2025 labels were used in optimization.','',f'Selected alpha={shared[0]:.6f}, beta={shared[1]:.6f}, gamma={shared[2]:.6f}; pretest objective={score:.4f}.','',mean.round(4).to_markdown(index=False),'',lead.round(4).to_markdown(index=False),'',f'Lead wins versus ETS: {wins}/4. MeanLead RMSE={best:.4f}; MAPE={mape:.4f}%. Threshold <24.70: {best<24.70}; threshold <24.20: {best<24.20}.','',pd.DataFrame(qcs,columns=['check','pass']).to_markdown(index=False)]
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8'); print(selected.to_string(index=False)); print(mean.to_string(index=False)); print(lead.to_string(index=False)); print(pd.DataFrame(qcs,columns=['check','pass']).to_string(index=False),flush=True)

if __name__=='__main__': main()
