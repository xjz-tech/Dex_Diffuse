import json,time,sys
from pathlib import Path
h=Path(__file__).resolve().parents[1]
if len(sys.argv)>1:h=Path(sys.argv[1])
methods=json.loads((h/'protocol.json').read_text())['methods']
for _ in range(45):
 pending=list((h/'runs').glob('*/physical_review_pending.json'))
 if pending:break
 time.sleep(1)
rows=[json.loads(p.read_text()) for p in (h/'runs').glob('*/summary.json')]
rows=[r for r in rows if r['phase']=='formal' and r['method'] in methods]
print('Completed:',len(rows),'/27')
for p in pending:print('REVIEW',p.parent.name,json.loads(p.read_text())['step'])
for p in (h/'runs').glob('*/pending.json'):
 if p.parent.name.startswith('screen') or p.parent.name.split('_e')[0] not in methods:continue
 if (p.parent/'summary.json').exists():continue
 print('ACTIVE/PENDING',p.parent.name)

for p in (h/"runs").glob("*/gait_events.json"):
 if not (p.parent/"summary.json").exists():
  a=json.loads(p.read_text())
  if a:print("LIVE",p.parent.name,"step",a[-1]["step"],"right",round(-a[-1]["twist_deg"],2))
