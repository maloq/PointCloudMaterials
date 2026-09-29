"""Interface-side readouts and exterior-only alarms on preserved incoming paths."""
import numpy as np

from src.research.spatial_distance.confidence import first_alarm
from .evaluate import RADII, point_tables


def phase_tables(values,meta,directional):
    tables=[[],[],[]]
    masks={'crystal_interior':meta['inside_crystal'] & ~meta['interface_member'],
           'interface_layer':meta['interface_member'],
           'outside_confirmed_crystal':~meta['inside_crystal'],
           'no_interface_in_cell':~meta['interface_exists']}
    for phase,mask in masks.items():
        if not mask.any():continue
        scores=point_tables({k:v[mask] for k,v in values.items()},
                            {k:v[mask] for k,v in meta.items()},directional)
        for result,rows in zip(tables,scores):result.extend(dict(phase=phase,**r) for r in rows)
    return tables


def interface_alarms(values,meta,plan):
    rows=[]
    for k,radius in enumerate(RADII):
        for threshold in (.5,.75,.95):
            toward=[];away=[];visible=0;eligible_lengths=[]
            for record in plan['paths']:
                if record['role']!='test':continue
                ids=np.flatnonzero(meta['path']==record['index'])
                ids=ids[np.argsort(meta['travel'][ids],kind='stable')]
                if len(ids)!=record['rows']:raise ValueError('Incomplete preserved scan')
                # These paths were designed to approach crystal, not to traverse its
                # interior. Stop BEFORE first entry; never count an exit-side alarm.
                entry=np.flatnonzero(meta['inside_crystal'][ids])
                eligible=ids[:int(entry[0])] if len(entry) else ids
                alarm=first_alarm(values['cdf'][eligible,k],threshold,2) if len(eligible)>=2 else None
                if record['kind']=='toward':
                    eligible_lengths.append(len(eligible))
                    toward.append(np.nan if alarm is None else meta['distance'][eligible[alarm]])
                    if alarm is not None:visible+=int(meta['visible_context'][eligible[alarm-1:alarm+1]].any())
                else:away.append(alarm is not None)
            a=np.asarray(toward);found=np.isfinite(a)
            rows.append(dict(radius_A=radius,threshold=threshold,consecutive=2,
                scan_scope='preserved exterior prefix before first crystal entry',toward_paths=len(a),
                paths_with_fewer_than_two_exterior_queries=int((np.asarray(eligible_lengths)<2).sum()),
                detections=int(found.sum()),misses=int((~found).sum()),
                conditional_median_warning_A=float(np.median(a[found])) if found.any() else None,
                recall_at12A=float((a>=12).mean()),recall_at20A=float((a>=20).mean()),
                away_paths=len(away),false_alarms=int(np.sum(away)),away_alarm_rate=float(np.mean(away)),
                interface_visible_at_alarm=visible))
    return rows
