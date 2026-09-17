"""Scan every stored image and deformation row; decode five camera samples per episode."""
import hashlib
import io
import json
import struct
from collections import Counter, defaultdict

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from analyze import ROOT, OUT, array, write_csv


def main():
    rows, errors = [], []
    deform_stats = defaultdict(list)
    digest_episodes = defaultdict(list)
    thumbs = OUT/'samples'
    thumbs.mkdir(exist_ok=True)
    for d in sorted((ROOT/'rank_0').glob('id_*'), key=lambda p:int(p.name[3:])):
        paths = list(d.glob('data/**/*.parquet'))
        if not paths: continue
        pf = pq.ParquetFile(paths[0])
        n = pf.metadata.num_rows
        sample_ids = set(np.linspace(0,n-1,5,dtype=int).tolist())
        canvases = {c:Image.new('RGB',(5*256,216),'#eef2f6') for c in ('image','extra_view_image')}
        draw = {c:ImageDraw.Draw(v) for c,v in canvases.items()}
        media = {c:Counter() for c in canvases}
        previous = {c:None for c in canvases}
        runs = {c:0 for c in canvases}
        peak_run = {c:0 for c in canvases}
        image_hashers = {c:hashlib.sha256() for c in canvases}
        offset = 0
        means, maxima, nonzero = [], [], []
        for rg in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg,columns=['image','extra_view_image','tactile_deform'])
            for c in canvases:
                for j,obj in enumerate(table[c].to_pylist()):
                    idx=offset+j
                    b=obj.get('bytes') if obj else None
                    if not b:
                        media[c]['empty']+=1; errors.append([d.name,c,idx,'empty bytes']); continue
                    h=hashlib.sha256(b).digest()
                    image_hashers[c].update(h)
                    repeated = h==previous[c]
                    media[c]['duplicate_adjacent'] += int(repeated)
                    runs[c] = runs[c]+1 if repeated else 1
                    peak_run[c]=max(peak_run[c],runs[c])
                    previous[c]=h
                    if b[:8]!=b'\x89PNG\r\n\x1a\n' or b[-8:]!=b'IEND\xaeB`\x82':
                        media[c]['bad_header_or_footer']+=1
                        errors.append([d.name,c,idx,'invalid PNG header/footer'])
                    else:
                        w,ht=struct.unpack('>II',b[16:24]);media[c][f'shape_{w}x{ht}']+=1
                    media[c]['frames']+=1
                    if idx in sample_ids:
                        try:
                            im=Image.open(io.BytesIO(b));im.load()
                            ar=np.asarray(im)
                            media[c]['decoded']+=1
                            media[c]['decoded_black']+=int(ar.max()==0)
                            if im.size != (640,480) or im.mode!='RGB': errors.append([d.name,c,idx,f'{im.size}/{im.mode}'])
                            col=sorted(sample_ids).index(idx)
                            canvases[c].paste(im.resize((256,192)),(col*256,24))
                            draw[c].text((col*256+5,6),f'{d.name} {c} f={idx}',fill='#152d44')
                        except Exception as e:
                            media[c]['decode_error']+=1;errors.append([d.name,c,idx,str(e)])
            deform=array(table,'tactile_deform')
            if deform.shape[1:] != (5,240,240): errors.append([d.name,'deform',offset,str(deform.shape)])
            means.append(deform.mean(axis=(2,3)))
            maxima.append(deform.max(axis=(2,3)))
            nonzero.append(np.count_nonzero(deform,axis=(2,3)))
            for idx in sorted(sample_ids):
                if offset <= idx < offset+len(deform) and int(d.name[3:]) in (0,13,35,47,72,75):
                    a=deform[idx-offset]
                    strip=Image.new('RGB',(5*240,270),'#eef2f6')
                    dr=ImageDraw.Draw(strip)
                    for s in range(5):
                        dr.text((s*240+8,7),f'{d.name} f={idx} sensor {s} max={a[s].max()}',fill='#152d44')
                        strip.paste(Image.fromarray(a[s]).convert('RGB'),(s*240,30))
                    strip.save(thumbs/f'{d.name}_deform_{idx}.png')
            offset+=table.num_rows
        for c,canvas in canvases.items(): canvas.save(thumbs/f'{d.name}_{c}.jpg',quality=88)
        means,maxima,nonzero=[np.concatenate(v) for v in [means,maxima,nonzero]]
        for k,v in [('mean',means),('max',maxima),('nonzero_pixels',nonzero),('episode_id',np.full(n,int(d.name[3:]))),('frame',np.arange(n))]:
            deform_stats[k].append(v)
        row={'id':d.name,'frames':n}
        for c in canvases:
            for k in ['frames','empty','bad_header_or_footer','duplicate_adjacent','decoded','decoded_black','decode_error']:
                row[c+'_'+k]=media[c][k]
            row[c+'_max_identical_run']=peak_run[c]
            digest_episodes[image_hashers[c].hexdigest()].append([d.name,c])
        for s in range(5):
            row[f'deform_{s}_nonzero_frames']=int((maxima[:,s]>0).sum())
            row[f'deform_{s}_max']=int(maxima[:,s].max())
        rows.append(row)
        print(f'media audit {len(rows)}/75: {d.name}',flush=True)
    write_csv(OUT/'media_episodes.csv',rows)
    da={k:np.concatenate(v) for k,v in deform_stats.items()}
    np.savez_compressed(OUT/'deform_arrays.npz',**da)
    summary={'scope':'All image bytes checked for PNG header/footer and adjacent equality; 5 images per camera per complete episode fully decoded. All deformation arrays scanned.',
             'totals':{k:sum(r[k] for r in rows) for k in rows[0] if k not in ['id'] and not k.endswith('_max') and not k.endswith('_max_identical_run')},
             'errors':errors,'duplicate_episode_camera_streams':[v for v in digest_episodes.values() if len(v)>1],
             'deform_nonzero_frame_fraction':(da['max']>0).mean(axis=0).tolist(),'deform_max':da['max'].max(axis=0).tolist(),
             'deform_all_sensors_zero_frames':int(np.all(da['max']==0,axis=1).sum()),
             'image_max_identical_run':max(r['image_max_identical_run'] for r in rows),
             'extra_view_image_max_identical_run':max(r['extra_view_image_max_identical_run'] for r in rows)}
    (OUT/'media_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
