"""Opt-in next-protocol collection: reuse AD values and preserve independent test seeds.

No historical archive is consumed as an output directory. Callers must supply a
new protocol identity containing the oracle implementation/backend/dtype settings.
Batch elapsed times are recorded as batch costs, never invented per-shot timings.
"""
import time
import shutil
from pathlib import Path
import torch
from src.experiment_runner.artifacts import write_json, file_hash as sha


def save(path, value):
    temporary = path.with_suffix('.building.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def acquire_parent(oracle, q, basis, *, role, value_seeds, response_seeds,
                   destination, identity, value_batch=4, response_batch=4,
                   audit_seeds=(), agreement_atol=1e-7, agreement_rtol=1e-5,
                   archive=None, deadline=float('inf')):
    if role not in ('train', 'selection', 'test'):
        raise ValueError(role)
    value_seeds, response_seeds = list(value_seeds), list(response_seeds)
    if len(set(value_seeds)) != len(value_seeds) or len(set(response_seeds)) != len(response_seeds):
        raise ValueError('Each trajectory bank requires unique seeds')
    if role == 'train' and value_seeds[:len(response_seeds)] != response_seeds:
        raise ValueError('Training response seeds must be the initial value seeds')
    if role == 'test' and set(value_seeds)&set(response_seeds):
        raise ValueError('Test value and response seeds must remain disjoint')
    if role == 'selection' and response_seeds:
        raise ValueError('Selection uses ordinary values only')
    if any(s not in response_seeds for s in audit_seeds) or role != 'train' and audit_seeds:
        raise ValueError('Execution audit must reuse declared training response seeds')
    if min(value_batch, response_batch) < 1:
        raise ValueError('Positive replica batches required')
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    durable = Path(archive) if archive is not None else None
    if durable is not None:
        durable.mkdir(parents=True, exist_ok=True)
        shutil.copytree(durable, root, dirs_exist_ok=True)

    def publish(path):
        if durable is not None:
            target = durable/path.name
            temporary = target.with_suffix(target.suffix+'.publishing')
            shutil.copy2(path, temporary)
            temporary.replace(target)
    contract = dict(protocol='response_reuse_collection_v1', identity=identity, role=role,
        value_seeds=value_seeds, response_seeds=response_seeds, audit_seeds=list(audit_seeds),
        value_batch=value_batch, response_batch=response_batch,
        dtype=str(q.dtype), agreement_atol=agreement_atol, agreement_rtol=agreement_rtol)
    path = root/'contract.pt'
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved['contract'] != contract or not torch.equal(saved['q'],q.cpu()) or not torch.equal(saved['basis'],basis.cpu()):
            raise ValueError('Collection protocol/input changed; use a new destination')
    else:
        if any(root.iterdir()):
            raise ValueError('New-protocol output must be empty; never append to a historical archive')
        save(path,dict(contract=contract,q=q.cpu(),basis=basis.cpu()))
    publish(path)
    costs = []

    def collect(kind, seeds, batch, directions):
        values, responses = {}, {}
        for start in range(0,len(seeds),batch):
            selected = seeds[start:start+batch]
            path = root/f'{kind}-{start:04d}.pt'
            if path.exists():
                result = torch.load(path,map_location='cpu',weights_only=False)
                if result['identity'] != identity or result['seeds'] != selected:
                    raise ValueError(f'Changed collected batch: {path}')
            else:
                if time.time() >= deadline:
                    raise TimeoutError('Allocation reserve reached; completed replica batches are archived')
                torch.cuda.synchronize(); began=time.monotonic()
                bundle=oracle.query(q,directions,selected)
                torch.cuda.synchronize()
                result=dict(identity=identity,seeds=selected,kind=kind,values=bundle.values.cpu(),
                    responses=bundle.responses.cpu(),batch_seconds=time.monotonic()-began,
                    logical_force_calls=bundle.force_calls,logical_hvp_calls=bundle.hvp_calls,
                    device=torch.cuda.get_device_name())
                save(path,result)
            publish(path)
            for i,seed in enumerate(selected):
                values[seed]=result['values'][i];responses[seed]=result['responses'][i]
            costs.append({k:result[k] for k in ('kind','seeds','batch_seconds','logical_force_calls','logical_hvp_calls','device')})
        return values,responses

    response_values,responses=collect('response',response_seeds,response_batch,basis)
    reused=response_values if role=='train' else {}
    ordinary=[s for s in value_seeds if s not in reused]
    ordinary_values,_=collect('value',ordinary,value_batch,q.new_empty(len(q),0))
    audit_values,_=collect('audit',list(audit_seeds),value_batch,q.new_empty(len(q),0))
    audit={}
    for seed in audit_seeds:
        error=float((audit_values[seed]-response_values[seed]).abs().max())
        audit[str(seed)]=dict(max_absolute=error)
        if not torch.allclose(audit_values[seed],response_values[seed],atol=agreement_atol,rtol=agreement_rtol):
            raise ValueError(f'Ordinary versus AD common-path audit failed: seed{seed}, error{error}')
    values={**ordinary_values,**reused}
    payload=dict(contract=contract,values=torch.stack([values[s] for s in value_seeds]),
        responses=torch.stack([responses[s] for s in response_seeds]) if response_seeds else
        torch.empty(0,256,basis.shape[-1],dtype=q.dtype),costs=costs,audit=audit,
        reused_value_seeds=list(reused),reference_note='Actual batch acquisition costs; value-only counterfactual requires its independent timing benchmark')
    save(root/'query.pt',payload)
    publish(root/'query.pt')
    write_json(root/'complete.json',dict(identity=identity,protocol=contract['protocol'],
        query_sha256=sha(root/'query.pt'),value_rows=len(value_seeds),response_rows=len(response_seeds),
        reused_value_rows=len(reused),audit_rows=len(audit_values),
        executed_trajectories=len(response_seeds)+len(ordinary)+len(audit_values),
        total_batch_seconds=sum(c['batch_seconds'] for c in costs)))
    publish(root/'complete.json')
    return payload
