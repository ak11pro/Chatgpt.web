"""Render a debugging image strip from RGB images and prediction CSV."""
import argparse,csv
from pathlib import Path
from PIL import Image,ImageDraw
def main():
 p=argparse.ArgumentParser();p.add_argument('--frames',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=12);a=p.parse_args(); rows=list(csv.DictReader(open(a.predictions)))[:a.limit]; tiles=[]
 for row in rows:
  image=Image.open(Path(a.frames)/f"{row['ID'].rsplit('_',1)[-1]}.jpg").convert('RGB');d=ImageDraw.Draw(image);x,y,w,h=map(int,(row['x'],row['y'],row['width'],row['height']));d.rectangle((x,y,x+w,y+h),outline='lime',width=2);tiles.append(image)
 width=max(x.width for x in tiles); strip=Image.new('RGB',(width,len(tiles)*tiles[0].height));[strip.paste(im,(0,i*im.height)) for i,im in enumerate(tiles)];strip.save(a.output)
if __name__=='__main__':main()
