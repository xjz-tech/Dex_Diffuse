import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,io
import numpy as np,pyarrow.parquet as pq
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent;rows=json.loads((OUT/'episodes.json').read_text());real=np.load(OUT/'real_states.npz')
ids=[0,14,29,44,59,74]
canvas=Image.new('RGB',(320*6,265*6),'white');draw=ImageDraw.Draw(canvas)
for row,e in enumerate(ids):
 meta=rows[e];table=pq.read_table(meta['path'],columns=['image'],use_threads=False);arr=table['image'];times=np.linspace(0,len(arr)-1,6).astype(int)
 for col,t in enumerate(times):
  im=Image.open(io.BytesIO(arr[int(t)].as_py()['bytes'])).convert('RGB');im.thumbnail((320,240));canvas.paste(im,(col*320,row*265+25))
  ind=np.flatnonzero(real['episode']==e)[t];ct=int((real['contact_count'][ind]>0).sum())
  draw.text((col*320+5,row*265+5),f"{meta['id']} frame {t} ({t/30:.1f}s) tactile fingers={ct}",fill='black')
canvas.save(OUT/'real_overview.jpg',quality=90);print(OUT/'real_overview.jpg')
