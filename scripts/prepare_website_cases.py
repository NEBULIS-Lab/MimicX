"""Export paper-selected Forest stages and common-reference comparison clips."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(project, ffmpeg, output, stages_only=False):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    def export(source, name, frame=None, crop=None):
        target = output/name
        args = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-hwaccel', 'none', '-i', str(source), '-map_metadata', '-1', '-an']
        if frame is not None:
            vf=f'select=eq(n\\,{frame})'
            if crop: vf+=f',crop={crop[2]-crop[0]}:{crop[3]-crop[1]}:{crop[0]}:{crop[1]}'
            args += ['-vf', vf+',scale=-2:720', '-frames:v', '1', '-c:v', 'libwebp', '-quality', '92', '-threads', '2']
        else:
            args += ['-vf','scale=-2:720','-c:v','libx264','-crf','25','-preset','medium','-threads','4','-movflags','+faststart']
        subprocess.run(args+[str(target)], check=True)
        records.append(dict(file=name, source=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), frame=frame, crop=crop))

    manifest=json.loads((project/'paper-repo-iclr/figures/revision/manifest.json').read_text())
    forest=next(r for r in manifest if r.get('figure')=='forest_stages_2x2')['source']
    for entry,name in zip(forest,['input','human','reference','policy']):
        export(Path(entry['source']),f'forest-stage-{name}.webp',entry['frame'],[398,160,1572,820] if name=='policy' else None)

    if stages_only:
        existing = json.loads((output/'manifest.json').read_text())
        records.extend(row for row in existing if not row['file'].startswith('forest-stage-'))
        (output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
        return

    data=list(csv.DictReader((project/'paper-repo-iclr/figures/direct_update/direct_update.csv').open()))
    public={}
    for task,key in [('tennis','tennis'),('football1','football')]:
        public[key]={}
        for method,label in [('m0_open_loop','fixed'),('BeyondMimic_MjLab','beyond'),('SONIC','sonic'),('m3_full_mimicx','ours')]:
            selected=[r for r in data if r['task']==task and r['method']==method]
            assert len(selected)==3 and len({r['reference_sha256'] for r in selected})==1
            row=next(r for r in selected if r['seed']=='202')
            source=Path(row['source']).with_name('rollout.mp4')
            export(source,f'direct-{key}-{label}.mp4')
            export(source,f'direct-{key}-{label}.webp',int(row['frames'])//2)
            public[key][label]=dict(body_mean=sum(float(r['root_local_fk_body_mean_m']) for r in selected)/3,
                                    frames=int(row['frames']), reference_sha256=row['reference_sha256'],display_seed=202)
    (output/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'comparison.json').write_text(json.dumps(public,indent=2)+'\n')
    return public


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'docs/assets/media/cases')
    parser.add_argument('--stages-only',action='store_true',help='Refresh stills while retaining the exported comparison recordings')
    args=parser.parse_args()
    prepare(args.project,args.ffmpeg,args.output,args.stages_only)
