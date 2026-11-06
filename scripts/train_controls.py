"""Train the matched controls on the 15 site-held-out splits of the main run."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
from dataclasses import replace, asdict
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'external_validation')]
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from dpi_evt.config import set_global_seed
from dpi_evt.data import prepare_benchmark_dataset
from dpi_evt.data.splits import make_grouped_cv_folds
from dpi_evt.experiments import final_run as F
from dpi_evt.evaluation import collect_outputs,fit_interval_scale
from dpi_evt.models.forecast_operators import make_transfer_class
from dpi_evt.models.controls import CrossingTransformer,DPINoReciprocalDrift
from dpi_evt.experiments.paths import primary_run, interval_run
BASE=primary_run(ROOT)
SPECS={
 'transformer_headP':dict(label='Transformer matched +P/head loss',curve_time_loss=False,anchor=False),
 'transformer_curveP':dict(label='Transformer matched +P/crossing loss',curve_time_loss=True,anchor=False),
 'transformer_curveA':dict(label='Transformer matched +A/crossing loss',curve_time_loss=True,anchor=True),
 'dpi_no_drift_A':dict(label='DPI-EVT +A/no reciprocal drift'),
}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(smoke=False):
 return dict(protocol='matched-controls-v1',base_run=str(BASE.relative_to(ROOT)),
  source_code=F.code_digest(ROOT),scripts={p.name:sha(p) for p in [Path(__file__),ROOT/'src/dpi_evt/models/controls.py']},
  models=SPECS,folds=15,seed_rule='42 + fold_index',width=160,max_epochs=1 if smoke else 200,
  patience=25,optimizer='AdamW',learning_rate=.002,weight_decay=.0001,batch_size=256,
  schedule='cosine',mc_samples=64,smoke=smoke,torch=torch.__version__,device='cpu',
  selection='minimum original internal validation loss; no tuning or seed selection',
  planned_contrasts=['matched curve versus matched head time supervision',
    'DPI-EVT versus matched projected Transformer','DPI-EVT+A versus matched anchored Transformer',
    'DPI-EVT+A versus same model with reciprocal drift removed'])
def output_dir(smoke=False):
 d=hashlib.sha256(json.dumps(identity(smoke),sort_keys=True).encode()).hexdigest()[:16]
 return ROOT/'results'/('control_smoke' if smoke else 'controls')/d

def benchmark(fi):
 pop,cfg=F.load_final_population(ROOT,10);cfg=F.effective_config(cfg,F.RunSettings())
 folds=make_grouped_cv_folds(pop['meta'],len(pop['meta']),42,cfg,n_splits=5,n_repeats=3,val_objects=2)
 fold=folds[fi];bm=prepare_benchmark_dataset(pop,cfg,torch.device('cpu'),precomputed_split=fold)
 for k in ['train','val','test']:F.attach_cap(bm[k])
 original=json.loads((BASE/f'primary/n0_10/fold_{fi}/dpi_evt_A/complete.json').read_text())
 for k in ['train','val','test']:assert bm[k]['meta'].specimen_id.tolist()==original[f'{k}_ids']
 return pop,replace(cfg,seed=42+fi),fold,bm

def make_model(key,fi,bm,cfg):
 if key=='dpi_no_drift_A':
  info=json.loads((BASE/f'primary/n0_10/fold_{fi}/dpi_evt_A/complete.json').read_text())
  kwargs=info['kwargs'];cls=make_transfer_class(DPINoReciprocalDrift)
 else:
  kwargs=dict(static_dim=bm['train']['static'].shape[1],seq_dim=bm['train']['seq_in'].shape[-1],
    seq_len=cfg.seq_len,hidden_dim=160,curve_time_loss=SPECS[key]['curve_time_loss'],
    anchor=SPECS[key]['anchor'],max_cycle_reference=cfg.max_cycle_reference)
  cls=CrossingTransformer
 set_global_seed(cfg.seed)
 return cls(**kwargs),kwargs

def external_splits(pop,bm,cfg,fi):
 from extval.final_run import ExternalRun
 from extval.common import build_external_population,make_external_split
 canonical={}
 for ds,recs in ExternalRun.load_records(pop['meta']).items():
  p,_=build_external_population(recs,cfg);canonical[ds]=p['meta'].specimen_id.tolist()
 parts={};hashes={}
 for ds,recs in ExternalRun.load_records(bm['train']['meta']).items():
  p,_=build_external_population(recs,cfg);p=F.subset_population(p,canonical[ds]);hashes[ds]=F.population_digest(p)
  parts[ds]=F.evaluation_split(F.attach_cap(make_external_split(p,bm['scalers'],cfg),external=True))
 expected=json.loads((BASE/f'external/primary/n0_10/fold_{fi}/inputs.json').read_text())
 assert hashes==expected,'External inputs differ from frozen analysis'
 return parts,hashes

def work(fi,smoke=False):
 started=time.monotonic();out=output_dir(smoke)
 pop,cfg,fold,bm=benchmark(fi)
 eval_cfg=replace(cfg,seed=42)
 ex,hashes=external_splits(pop,bm,eval_cfg,fi)
 F.write_json(out/f'fold_{fi}/inputs.json',dict(external=hashes,
  ids={k:bm[k]['meta'].specimen_id.tolist() for k in ['train','val','test']}))
 for key,spec in SPECS.items():
  dest=out/f'fold_{fi}'/key;marker=dest/'complete.json'
  if marker.exists():
   info=json.loads(marker.read_text())
   assert all(sha(dest/name)==digest for name,digest in info['outputs'].items())
   print(f'CACHED fold={fi} {key}',flush=True);continue
  dest.mkdir(parents=True,exist_ok=True)
  model,kwargs=make_model(key,fi,bm,cfg)
  print(f'TRAIN fold={fi} {key}',flush=True);t=time.monotonic()
  model,history=F.train_model(model,bm['train'],bm['val'],epochs=1 if smoke else 200,
    model_name=spec['label'],config=cfg,device=torch.device('cpu'),verbose=False,scheduler='cosine',ema_decay=0.)
  fit_interval_scale(model,bm['val'],cfg,torch.device('cpu'),level=.9)
  history.to_csv(dest/'history.csv',index=False);torch.save(model.state_dict(),dest/'weights.pt')
  metrics=[];frames=[]
  for source,split in [('internal',bm['test']),*ex.items()]:
   outputs=collect_outputs(model,split,eval_cfg,torch.device('cpu'))
   F.write_npz(dest/f'{source}.npz',outputs)
   met,rows=F.evaluate_outputs(spec['label'],outputs,split,bm['train'],eval_cfg)
   metrics.append(dict(source=source,model=spec['label'],key=key,fold=fi,repeat=int(fold['repeat']),**met))
   frames.append(rows.assign(source=source,model=spec['label'],key=key,fold=fi,repeat=int(fold['repeat'])))
  val=collect_outputs(model,bm['val'],eval_cfg,torch.device('cpu'));F.write_npz(dest/'validation.npz',val)
  pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
  pd.concat(frames,ignore_index=True).to_csv(dest/'per_test.csv',index=False)
  info=dict(key=key,fold=fi,repeat=int(fold['repeat']),seed=cfg.seed,kwargs=kwargs,epochs=len(history),
    best_epoch=int(history.loc[history.val_loss.idxmin(),'epoch']),parameter_count=sum(p.numel() for p in model.parameters()),
    training_seconds=time.monotonic()-t,outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'})
  F.write_json(marker,info)
  print(f'DONE fold={fi} {key} epochs={len(history)} seconds={time.monotonic()-t:.1f}',flush=True)
 return dict(fold=fi,seconds=time.monotonic()-started)

def run_controls(workers=3,smoke=False):
 out=output_dir(smoke);out.mkdir(parents=True,exist_ok=True)
 wanted=identity(smoke);manifest=out/'manifest.json'
 if manifest.exists():assert json.loads(manifest.read_text())==wanted
 else:F.write_json(manifest,wanted)
 folds=[0] if smoke else list(range(15))
 missing=[fi for fi in folds if not all((out/f'fold_{fi}'/k/'complete.json').exists() for k in SPECS)]
 if missing:
  with ProcessPoolExecutor(max_workers=workers) as pool:
   jobs={pool.submit(work,fi,smoke):fi for fi in missing}
   for fut in as_completed(jobs):print('FOLD COMPLETE',fut.result(),flush=True)
 metrics=[];records=[];hist=[]
 for fi in folds:
  for key in SPECS:
   dest=out/f'fold_{fi}'/key;info=json.loads((dest/'complete.json').read_text())
   assert all(sha(dest/name)==v for name,v in info['outputs'].items())
   metrics.append(pd.read_csv(dest/'metrics.csv'));records.append(pd.read_csv(dest/'per_test.csv',low_memory=False))
   hist.append({k:v for k,v in info.items() if k not in ['outputs','kwargs']})
 pd.concat(metrics,ignore_index=True).to_csv(out/'metrics.csv',index=False)
 pd.concat(records,ignore_index=True).to_csv(out/'per_test.csv',index=False)
 pd.DataFrame(hist).to_csv(out/'training_summary.csv',index=False)
 F.write_json(out/'completed.json',dict(status='complete',models=len(hist),folds=len(folds),
  outputs={p.name:sha(p) for p in [out/'metrics.csv',out/'per_test.csv',out/'training_summary.csv']}))
 print('COMPLETE',out,flush=True)
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--smoke',action='store_true')
 a=p.parse_args();run_controls(a.workers,a.smoke)
