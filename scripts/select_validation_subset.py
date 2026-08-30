import argparse, csv, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hotc_tracker.data.sequences import group_annotations
def main():
 p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--seed',type=int,default=2026);a=p.parse_args(); g=group_annotations(a.annotations); buckets={}
 for key,rows in g.items():
  boxes=[float(r['width'])*float(r['height']) for _,r in rows]; bucket=(rows[0][0].sensor, len(rows)//100, int(sum(boxes)/len(boxes)//1000)); buckets.setdefault(bucket,[]).append(key)
 rng=random.Random(a.seed); chosen=[]
 for b in sorted(buckets): rng.shuffle(buckets[b]);chosen.append(buckets[b][0])
 rest=[x for xs in buckets.values() for x in xs if x not in chosen];rng.shuffle(rest);chosen=(chosen+rest)[:a.count];Path(a.output).write_text('\n'.join(chosen)+'\n')
if __name__=='__main__': main()
