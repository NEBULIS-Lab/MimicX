"""Scene continuation gates and practice windows from the recorded terrain run."""
import numpy as np


def failed(row):return str(row['done_any']).lower() in ('true','1')


def summarize(rows,horizon):
    if len(rows)!=horizon or [int(r['step']) for r in rows]!=list(range(1,horizon+1)):
        raise ValueError('Incomplete/nonconsecutive rollout')
    body=np.array([float(r['body_pos_error_max']) for r in rows])
    root=np.array([[float(r[f'robot_anchor_{a}'])-float(r[f'ref_anchor_{a}']) for a in 'xyz'] for r in rows])
    if not np.isfinite(body).all() or not np.isfinite(root).all():raise ValueError('Nonfinite rollout')
    frontier=next((i for i,r in enumerate(rows) if failed(r)),horizon)
    root=np.linalg.norm(root,axis=1)
    return dict(valid_steps=frontier,complete=frontier==horizon,resets=sum(map(failed,rows)),
        body_p95=float(np.quantile(body[:frontier],.95)) if frontier else None,
        root_p95=float(np.quantile(root[:frontier],.95)) if frontier else None)


def prefix(rows,n):
    if not n:return None
    return [float(np.quantile([float(r['body_pos_error_max']) for r in rows[:n]],.95)),
            float(np.quantile([np.linalg.norm([float(r[f'robot_anchor_{a}'])-float(r[f'ref_anchor_{a}'])
                                              for a in 'xyz']) for r in rows[:n]],.95))]


def decision(old,new,horizon,require_improvement=True):
    if len(old)!=len(new) or not old:raise ValueError('Paired repeats are required')
    a=[summarize(r,horizon) for r in old];b=[summarize(r,horizon) for r in new]
    pairs=[(prefix(x,min(sa['valid_steps'],sb['valid_steps'])),
            prefix(y,min(sa['valid_steps'],sb['valid_steps']))) for x,y,sa,sb in zip(old,new,a,b)]
    finite=all(x is not None and y is not None for x,y in pairs)
    guards=finite and all(all(yv<=xv*1.10+.005 for xv,yv in zip(x,y)) for x,y in pairs)
    no_regression=all(y['valid_steps']>=x['valid_steps'] for x,y in zip(a,b))
    rank=lambda reports:(sum(x['complete'] for x in reports),min(x['valid_steps'] for x in reports),
                          sum(x['valid_steps'] for x in reports))
    quality=finite and np.mean([sum(y) for x,y in pairs])<.98*np.mean([sum(x) for x,y in pairs])
    improvement=rank(b)>rank(a) or (rank(b)==rank(a) and quality)
    return dict(accepted=bool(guards and no_regression and (improvement or not require_improvement)),
        old=a,new=b,paired_common_prefix_p95=pairs,tracking_guards=bool(guards),
        no_seed_frontier_regression=no_regression,execution_or_quality_improved=bool(improvement),
        uses_training_reward=False)


def diagnose(repeats,horizon):
    summaries=[summarize(r,horizon) for r in repeats]
    frontier=min(r['valid_steps'] for r in summaries)
    if frontier==0:return dict(frontier=0,peak_frame=0,summaries=summaries)
    length=max(1,frontier)
    # Terminal samples may already contain a reset pose.
    scores=np.mean([[float(r['body_pos_error_max']) for r in rows[:length]] for rows in repeats],axis=0)
    return dict(frontier=frontier,peak_frame=int(np.argmax(scores)),summaries=summaries)


def practice_windows(route,diag,horizon,events):
    clamp=lambda a,b,p:[max(0,min(int(a),horizon-1)),max(1,min(int(b),horizon)),p]
    frontier,peak=diag['frontier'],diag['peak_frame']
    if route=='world_fidelity':windows=[clamp(peak-25,peak+30,.25),[0,horizon,.15]]
    elif route=='event_transition':
        event=events.get('landing',[max(0,frontier-40),frontier])
        windows=[clamp(event[0]-25,horizon,.40),clamp(peak-20,peak+25,.20)]
    else:windows=[clamp(frontier-40,frontier+25,.35),clamp(peak-20,peak+25,.15),[0,horizon,.10]]
    for w in windows:w[1]=max(w[0]+1,w[1])
    return windows


def configured_practice_windows(config,diagnosis,block):
    if config.get('curriculum')=='tail_consolidation':
        from .tail_objectives import tail_windows
        return tail_windows(config,block)
    if config.get('curriculum')=='landing_roll_bridge':
        from .phase_dynamics import bridge_windows
        return bridge_windows(config,block)
    return practice_windows(config.get('objective_route',config['route']),diagnosis,
                            config['horizon'],config['events'])


def configured_decision(config,old,new,require_improvement=True):
    if config.get('tail_quality_gate',False):
        from .tail_objectives import tail_decision
        return tail_decision(old,new,config,require_improvement)
    return decision(old,new,config['horizon'],require_improvement)
