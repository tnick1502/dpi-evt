"""Compare matched controls, drift ablation and Transformer/GBM reconciliations."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'external_validation')]
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from train_controls import BASE,output_dir,benchmark,external_splits,SPECS,sha
from dpi_evt.experiments import final_run as F
from dpi_evt.models.forecast_operators import anchor_and_cap
from dpi_evt.training.losses import interpolated_crossing
B=20000
SEED=441105
A='DPI-EVT +A';TR='Transformer +cummax'


def reconcile_curve(transformer,gbm,split):
    """Fixed reconciliation that uses only the predictions and the observed prefix.

    Anchor the projected curve to observed pressure. For a GBM finite median,
    rescale a non-crossing continuation's remaining amplitude to 1 at H, then
    time-warp the post-prefix curve so its continuous crossing targets that
    median. For a beyond-H median, compress the unobserved increment below .95.
    Resample on the original grid and read the point time from the resampled curve.
    The GBM time itself is not reported as a curve crossing.
    """
    r=anchor_and_cap(torch.as_tensor(transformer['traj_mean']),split,anchor=True,cap=False).numpy()
    cyc=split['cycles'].numpy();pm=split['prefix_mask'].numpy()>.5
    idx=np.maximum(np.where(pm,np.arange(r.shape[1]),-1).max(1),0)
    ro=r[np.arange(len(r)),idx];n0=10.;H=3000.;tau=.95
    point=F.resolve_nliq_prediction(gbm,H)
    finite=np.asarray(gbm['event_tail_mass'])<=.5
    result=r.copy()
    for i in range(len(r)):
        future=cyc[i]>n0
        if not finite[i]:
            span=max(float(r[i,-1]-ro[i]),1e-6)
            ceiling=max(float(ro[i]),tau-1e-4)
            result[i,future]=ro[i]+np.clip((r[i,future]-ro[i])/span,0,1)*(ceiling-ro[i])
            continue
        work=r[i].copy()
        if work[-1]<tau:
            span=float(work[-1]-ro[i])
            if span>1e-6:work[future]=ro[i]+(1-ro[i])*(work[future]-ro[i])/span
            else:work[future]=ro[i]+(1-ro[i])*(cyc[i,future]-n0)/(H-n0)
        old=float(interpolated_crossing(torch.tensor(work[None],dtype=torch.float32),torch.tensor(cyc[i:i+1]),tau).item())
        goal=float(np.clip(point[i],n0+1e-3,H))
        warped=n0+(cyc[i,future]-n0)*max(old-n0,1e-3)/(goal-n0)
        result[i,future]=np.interp(warped,cyc[i],work)
    result=np.maximum.accumulate(np.clip(result,0,1),axis=1).astype(np.float32)
    actual=interpolated_crossing(torch.tensor(result),torch.tensor(cyc),tau).numpy()
    return dict(traj_mean=result,nliq=actual,risk_prob=gbm['risk_prob'])


def threshold_projected_hybrid(transformer,gbm,split):
    """Threshold-constrained pressure projection of the Transformer forecast.

    Keep the measured prefix. Clip the forecast before the predicted event and
    floor it after the event. Set the two bracketing query values at .95 with
    a fixed total pressure margin 1e-4 and the target interpolation fraction.
    This introduces near-threshold plateaus, which remain visible in examples.
    If the event lies in the first future grid interval, retain the prefix value
    and cap the next value at 1, then report the crossing of the resulting curve.
    Observed continuations and event labels are not used.
    """
    r=anchor_and_cap(torch.as_tensor(transformer['traj_mean']),split,anchor=True,cap=False).numpy().astype(float)
    cycles=split['cycles'].numpy();pm=split['prefix_mask'].numpy()>.5
    last=np.maximum(np.where(pm,np.arange(r.shape[1]),-1).max(1),0)
    goal=F.resolve_nliq_prediction(gbm,3000.);finite=np.asarray(gbm['event_tail_mass'])<=.5
    tau=.95;eps=1e-4
    for i in range(len(r)):
        j=last[i];ro=r[i,j]
        if ro>=tau:continue  # The measured prefix already implies a curve crossing.
        if not finite[i]:r[i,j+1:]=np.minimum(r[i,j+1:],tau-eps);continue
        t=float(np.clip(goal[i],10.+1e-3,3000.))
        k=int(np.clip(np.searchsorted(cycles[i],t,side='left')-1,j,len(r[i])-2))
        w=float(np.clip((t-cycles[i,k])/(cycles[i,k+1]-cycles[i,k]),1e-9,1.))
        if k==j:
            r[i,k+1]=min(1.,ro+(tau-ro)/w)
            r[i,k+2:]=np.maximum(r[i,k+2:],r[i,k+1])
        else:
            low=max(ro,tau-eps*w);high=tau+eps*(1-w)
            r[i,j+1:k]=np.minimum(r[i,j+1:k],low)
            r[i,k]=low;r[i,k+1]=high
            r[i,k+2:]=np.maximum(r[i,k+2:],high)
    r=np.maximum.accumulate(np.clip(r,0,1),axis=1)
    actual=interpolated_crossing(torch.tensor(r),torch.tensor(cycles),tau).numpy()
    return dict(traj_mean=r,nliq=actual,risk_prob=gbm['risk_prob'])


def build_hybrid(out):
    dest=out/'hybrid';dest.mkdir(parents=True,exist_ok=True)
    definition={'protocol':'anchored-timewarp-hybrid-v1','training':False,'uses_outcomes':False,
      'inputs':'Frozen Transformer and GBM predictions plus observed prefix',
      'point_readout':'Actual interpolated crossing of final original-grid curve',
      'definition':reconcile_curve.__doc__,
      'threshold_projection':threshold_projected_hybrid.__doc__,'script_sha256':sha(__file__)}
    # Recompute when the reconciliation code or its description changes.
    old=dest/'definition.json'
    if old.exists() and json.loads(old.read_text())==definition and (dest/'per_test.csv').exists():
        return pd.read_csv(dest/'per_test.csv',low_memory=False)
    F.write_json(old,definition)
    frames=[];metrics=[]
    for fi in range(15):
        pop,cfg,fold,bm=benchmark(fi);cfg=__import__('dataclasses').replace(cfg,seed=42)
        ex,_=external_splits(pop,bm,cfg,fi)
        for ds,split in [('internal',bm['test']),*ex.items()]:
            path=BASE/f'primary/n0_10/fold_{fi}' if ds=='internal' else BASE/f'external/primary/n0_10/fold_{fi}'
            trfile=path/'transformer'/('predictions.npz' if ds=='internal' else f'{ds}.npz')
            gbfile=path/'gbm_distribution'/('predictions.npz' if ds=='internal' else f'{ds}.npz')
            with np.load(trfile) as z:tr=dict(z)
            with np.load(gbfile) as z:gb=dict(z)
            for name,func,key in [('Time-warped Transformer/GBM',reconcile_curve,'timewarp'),
                                  ('Threshold-projected Transformer/GBM',threshold_projected_hybrid,'projection')]:
                outputs=func(tr,gb,split)
                F.write_npz(dest/f'fold_{fi}'/f'{ds}_{key}.npz',outputs)
                met,rows=F.evaluate_outputs(name,outputs,split,bm['train'],cfg)
                frames.append(rows.assign(source=ds,model=name,fold=fi,repeat=int(fold['repeat'])))
                metrics.append(dict(source=ds,model=name,fold=fi,repeat=int(fold['repeat']),**met))
    records=pd.concat(frames,ignore_index=True);records.to_csv(dest/'per_test.csv',index=False)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    return records


def pooled(g,source):
    unit='repeat' if source=='internal' else 'fold'
    curves=[];onsets=[]
    for _,x in g.groupby(unit):
        valid=x.traj_sse_continuation.notna()
        curves.append(np.sqrt(x.loc[valid,'traj_sse_continuation'].sum()/x.loc[valid,'continuation_points'].sum()) if valid.any() else np.nan)
        e=x[x.liq_label==1];onsets.append(np.abs(np.log1p(e.nliq_pred)-np.log1p(e.N_liq_true)).mean())
    ev=g[g.liq_label==1]
    return dict(tests=g.specimen_id.nunique(),events=ev.specimen_id.nunique(),
        curve_tests=g.loc[g.traj_sse_continuation.notna() & g.continuation_points.gt(0),'specimen_id'].nunique(),
        curve_rmse=np.mean(curves),onset_logmae=np.mean(onsets),
        noncrossing=ev.curve_crosses_threshold.eq(0).mean() if g.traj_sse_continuation.notna().any() else np.nan)


def site_comparisons(internal,pairs):
    sites=sorted(internal.site_id.unique());rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(sites),(B,len(sites)))
    w=np.stack([(draws==i).sum(1) for i in range(len(sites))],axis=1)
    distributions={};siteerr={};results=[]
    for model,g in internal.groupby('model'):
        event=g[g.liq_label==1].copy()
        event['error']=np.abs(np.log1p(event.nliq_pred)-np.log1p(event.N_liq_true))
        per=event.groupby('site_id').error.agg(['sum','count']).reindex(sites).fillna(0)
        timing=(w@per['sum'].values)/np.maximum(w@per['count'].values,1)
        curve=[]
        for _,x in g.groupby('repeat'):
            ag=x.groupby('site_id').agg(sse=('traj_sse_continuation',lambda v:v.sum(min_count=1)),n=('continuation_points','sum')).reindex(sites)
            curve.append(np.sqrt((w@ag.sse.values)/(w@ag.n.values)))
        distributions[model]={'onset_logmae':timing,'curve_rmse':np.mean(curve,axis=0)}
        siteerr[model]=per['sum']/per['count'].replace(0,np.nan)
    for a,b in pairs:
        if a not in distributions or b not in distributions:continue
        aa=pooled(internal[internal.model==a],'internal');bb=pooled(internal[internal.model==b],'internal')
        for metric in ['onset_logmae','curve_rmse']:
            if not np.isfinite(aa[metric]+bb[metric]):continue
            av=distributions[a][metric];bv=distributions[b][metric];ratio=av/bv
            delta=av-bv
            results.append(dict(reference=a,comparator=b,metric=metric,reference_value=aa[metric],comparator_value=bb[metric],
                ratio=aa[metric]/bb[metric],ratio_lo=np.quantile(ratio,.025),ratio_hi=np.quantile(ratio,.975),
                delta=aa[metric]-bb[metric],delta_lo=np.quantile(delta,.025),delta_hi=np.quantile(delta,.975),
                bootstrap_fraction_ratio_ge_1=(ratio>=1).mean(),sites_better=int((siteerr[a]<siteerr[b]).sum()) if metric=='onset_logmae' else np.nan,
                sites=len(sites),draws=B,seed=SEED))
    return pd.DataFrame(results)


def bootstrap_curve_rmse(sse, points, weights):
    """Pooled continuation RMSE for bootstrap draws of tests within a source.

    Records without continuation points add nothing to the error or the denominator,
    but stay in the resampling population. Missing SSE with a positive point count
    raises an error.
    """
    sse=np.asarray(sse,dtype=float);points=np.asarray(points,dtype=float)
    if not np.isfinite(points).all() or (points<0).any():
        raise ValueError('Continuation point counts must be finite and nonnegative')
    if ((~np.isfinite(sse)) & (points>0)).any():
        raise ValueError('Nonfinite continuation SSE with positive point count')
    if (sse[np.isfinite(sse)]<0).any():
        raise ValueError('Continuation SSE must be nonnegative')
    numerator=weights@np.where(points>0,sse,0.)
    denominator=weights@points
    return np.sqrt(np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0))


def external_comparisons(records,pairs):
    rows=[];rng=np.random.default_rng(SEED+1)
    for source,d in records[records.source!='internal'].groupby('source'):
        models=set(d.model)
        ids=sorted(d.specimen_id.unique());events=sorted(d[d.liq_label==1].specimen_id.unique())
        def weights(n):
            draws=rng.integers(0,n,(B,n));return np.stack([(draws==i).sum(1) for i in range(n)],axis=1)
        wt=weights(len(ids));we=weights(len(events))
        boot={};est={}
        for m,g in d.groupby('model'):
            est[m]=pooled(g,source);cv=[];te=[]
            for _,x in g.groupby('fold'):
                t=x.set_index('specimen_id').reindex(ids)
                if t.traj_sse_continuation.notna().any():
                    cv.append(bootstrap_curve_rmse(t.traj_sse_continuation.values,t.continuation_points.values,wt))
                ev=t.reindex(events);err=np.abs(np.log1p(ev.nliq_pred)-np.log1p(ev.N_liq_true)).values
                te.append((we@err)/len(events))
            boot[m]={'curve_rmse':np.mean(cv,axis=0) if cv else np.full(B,np.nan),'onset_logmae':np.mean(te,axis=0)}
        for a,b in pairs:
            if a not in models or b not in models:continue
            for metric in ['curve_rmse','onset_logmae']:
                if not np.isfinite(est[a][metric]+est[b][metric]):continue
                v=boot[a][metric]/boot[b][metric];finite=v[np.isfinite(v)]
                ci=np.quantile(finite,[.025,.975]) if len(finite) else np.full(2,np.nan)
                if metric=='onset_logmae' and len(events)<2:ci[:]=np.nan
                rows.append(dict(source=source,reference=a,comparator=b,metric=metric,ratio=est[a][metric]/est[b][metric],
                    ratio_lo=ci[0],ratio_hi=ci[1],tests=len(ids),events=len(events),
                    curve_tests_reference=est[a]['curve_tests'],curve_tests_comparator=est[b]['curve_tests'],
                    valid_draws=len(finite),
                    ci_unit='test within source, all 15 predictions retained',draws=B))
    return pd.DataFrame(rows)


def sensitivity(internal):
    pop,_=F.load_final_population(ROOT,10)
    from extval.cohort import ru_cap
    meta=pop['meta'].copy();cap=ru_cap(meta,c_to_kpa=1000.)
    maximum=np.where(pop['valid_mask']>0,pop['r_obs'],-np.inf).max(1)
    meta['exceeds_cap']=maximum>cap+1e-9;meta['event']=pop['liq_label']>.5
    meta[['specimen_id','soil_type','event','exceeds_cap']].to_csv(output_dir()/'audit/cap_sensitivity_membership.csv',index=False)
    pd.crosstab(meta[meta.event].soil_type,meta[meta.event].exceeds_cap).to_csv(output_dir()/'audit/cap_event_soils.csv')
    rows=[]
    for label,ids in [('all_primary',set(meta.specimen_id)),('does_not_exceed_proposed_cap',set(meta[~meta.exceeds_cap].specimen_id)),
                       ('exceeds_proposed_cap',set(meta[meta.exceeds_cap].specimen_id)),
                       ('all_sands',set(meta[meta.soil_type.str.contains('sand') & ~meta.soil_type.eq('sandy_loam')].specimen_id)),
                       ('sands_not_exceeding_cap',set(meta[meta.soil_type.str.contains('sand') & ~meta.soil_type.eq('sandy_loam') & ~meta.exceeds_cap].specimen_id))]:
        for model,g in internal[internal.specimen_id.isin(ids)].groupby('model'):
            rows.append(dict(subset=label,model=model,**pooled(g,'internal')))
    return pd.DataFrame(rows)


def control_intervals(out,records):
    """Apply the same fixed validation-residual method to all six new references."""
    from calibrate_intervals import OnsetCalibrator
    from dpi_evt.evaluation import collect_outputs
    from dpi_evt.models.forecast_operators import CalibratedTransformer
    import joblib
    from dataclasses import replace
    rows=[];parameters={}
    for fi in range(15):
        _,cfg,_,bm=benchmark(fi);cfg=replace(cfg,seed=42)
        val=bm['val'];ids=val['meta'].specimen_id.tolist()
        outputs={}
        for key,spec in SPECS.items():
            with np.load(out/f'fold_{fi}/{key}/validation.npz') as z:outputs[key]=(spec['label'],dict(z))
        path=BASE/f'primary/n0_10/fold_{fi}'
        info=json.loads((path/'transformer/complete.json').read_text())
        tr=CalibratedTransformer(**info['kwargs'])
        tr.load_state_dict(torch.load(path/'transformer/weights.pt',map_location='cpu',weights_only=True))
        tr=collect_outputs(tr,val,cfg,torch.device('cpu'))
        gb=collect_outputs(joblib.load(path/'gbm_distribution/estimator.joblib'),val,cfg,torch.device('cpu'))
        outputs['hybrid_timewarp']=('Time-warped Transformer/GBM',reconcile_curve(tr,gb,val))
        outputs['hybrid_projection']=('Threshold-projected Transformer/GBM',threshold_projected_hybrid(tr,gb,val))
        for key,(label,pred) in outputs.items():
            point=F.resolve_nliq_prediction(pred,3000.)
            for target in [.9,.95]:
                calibrator=OnsetCalibrator.fit(val['n_liq_true'].numpy(),point,val['label'].numpy(),target)
                parameters[f'{key}/{fi}/{target}']=dict(validation_ids=ids,**calibrator.parameters())
                data=records[(records.model==label)&(records.fold==fi)&(records.liq_label==1)]
                lo,hi=calibrator.predict(data.nliq_pred.to_numpy())
                for item,l,h in zip(data[['source','specimen_id','N_liq_true']].to_dict('records'),lo,hi):
                    rows.append(dict(model=key,label=label,fold=fi,target=target,**item,lower=l,upper=h,
                        covered=l<=item['N_liq_true']<=h,width_H=min(h,3000.)-min(l,3000.)))
    intervals=pd.DataFrame(rows);intervals.to_csv(out/'audit/control_intervals.csv',index=False)
    F.write_json(out/'audit/control_interval_parameters.json',parameters)
    summary=intervals.groupby(['model','label','source','target']).agg(
        coverage=('covered','mean'),mean_width_H=('width_H','mean'),median_width_H=('width_H','median'),
        event_forecasts=('covered','size'),unique_events=('specimen_id','nunique')).reset_index()
    summary.to_csv(out/'audit/control_interval_summary.csv',index=False)
    return summary


def analyze(require_controls=True):
    out=output_dir();dest=out/'audit';dest.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(BASE/'primary/n0_10/per_test.csv',low_memory=False).assign(source='internal')
    e=pd.read_csv(BASE/'external/primary/n0_10/per_test.csv',low_memory=False).rename(columns={'dataset':'source'})
    hybrid=build_hybrid(out)
    frames=[p,e,hybrid]
    if (out/'completed.json').exists():frames.append(pd.read_csv(out/'per_test.csv',low_memory=False))
    elif require_controls:raise RuntimeError('Control training is not complete')
    records=pd.concat(frames,ignore_index=True)
    assert not records.duplicated(['source','model','fold','specimen_id']).any()
    rows=[dict(source=source,model=model,**pooled(g,source)) for (source,model),g in records.groupby(['source','model'])]
    summary=pd.DataFrame(rows);summary.to_csv(dest/'joint_metrics.csv',index=False)
    internal=records[records.source=='internal']
    pairs=[(A,TR),('DPI-EVT',TR),(A,'EVT-NeuralSSM'),(A,'Transformer +A'),
       (A,'Time-warped Transformer/GBM'),(A,'Threshold-projected Transformer/GBM'),(A,SPECS['transformer_curveA']['label']),
       ('DPI-EVT',SPECS['transformer_curveP']['label']),
       (SPECS['transformer_curveP']['label'],SPECS['transformer_headP']['label']),
       (A,SPECS['dpi_no_drift_A']['label'])]
    comparisons=site_comparisons(internal,pairs);comparisons.to_csv(dest/'internal_paired_bootstrap.csv',index=False)
    external_comparisons(records,pairs).to_csv(dest/'external_paired_bootstrap.csv',index=False)
    sensitivity(internal).to_csv(dest/'cap_sensitivity.csv',index=False)
    # Leave out one site at a time with all its repeats to measure site influence.
    influence=[]
    for site in sorted(internal.site_id.unique()):
        d=internal[internal.site_id!=site]
        for a,b in pairs:
            if a not in set(d.model) or b not in set(d.model):continue
            am=pooled(d[d.model==a],'internal');bm=pooled(d[d.model==b],'internal')
            influence.append(dict(omitted_site=site,reference=a,comparator=b,onset_ratio=am['onset_logmae']/bm['onset_logmae']))
    pd.DataFrame(influence).to_csv(dest/'leave_one_site_out.csv',index=False)
    from dpi_evt.experiments.paths import interval_run
    cal=interval_run(ROOT)
    native=pd.read_csv(cal/'external_summary.csv');native=native[(native.method=='validation_log_asymmetric')&np.isclose(native.target,.9)]
    labels=pd.read_csv(cal/'model_inventory.csv').set_index('model').label
    native['label']=native.model.map(labels)
    if (out/'completed.json').exists():
        extra=control_intervals(out,records)
        native=pd.concat([native,extra[(extra.source!='internal')&np.isclose(extra.target,.9)]],ignore_index=True)
    native.to_csv(dest/'external_interval_comparison.csv',index=False)
    sources=[BASE/'primary/n0_10/per_test.csv',BASE/'external/primary/n0_10/per_test.csv',cal/'external_summary.csv',Path(__file__),ROOT/'src/dpi_evt/models/controls.py']
    if (out/'completed.json').exists():sources.append(out/'per_test.csv')
    F.write_json(dest/'analysis_manifest.json',dict(status='complete' if (out/'completed.json').exists() else 'frozen_only',
       bootstrap_draws=B,seed=SEED,sites=13,scope='Existing cohorts and predefined bounded controls; no outcome-based exclusion of the primary cohort',
       internal_ci='paired site bootstrap, frozen OOF predictions, all repeats retained; empirical sensitivity, not a null-hypothesis p-value',
       external_ci='paired test bootstrap conditional on source and frozen fits; one-event timing CI unavailable',
       cap_subset='exploratory outcome-defined sensitivity to proposed cap; not ground truth or corrected primary labels',
       sources={str(v.relative_to(ROOT)):sha(v) for v in sources}))
    print(summary[summary.source=='internal'][['model','curve_rmse','onset_logmae']].to_string(index=False))
    print(comparisons.to_string(index=False));return dest


def ensure_analysis():
    out=output_dir();dest=out/'audit';path=dest/'analysis_manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        if m['status']=='complete' and all((ROOT/p).exists() and sha(ROOT/p)==v for p,v in m['sources'].items()):
            return dest
    return analyze()
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--frozen-only',action='store_true');a=p.parse_args()
    analyze(require_controls=not a.frozen_only)
