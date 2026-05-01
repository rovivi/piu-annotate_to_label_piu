"""
Copies processed_db JSONs to piulatam/public/chart-jsons/{chart_id}.json
Run after process_db_matches.py finishes.
"""
import os, json, shutil

src = 'artifacts/processed_db'
dst = '/home/rodrigo/dev/piu/piulatam/public/chart-jsons'

os.makedirs(dst, exist_ok=True)
copied = skipped = 0

for fname in os.listdir(src):
    if not fname.endswith('.json'):
        continue
    parts = fname[:-5].rsplit('_', 1)
    if len(parts) != 2:
        skipped += 1
        continue
    chart_id = parts[1]
    shutil.copy2(os.path.join(src, fname), os.path.join(dst, f'{chart_id}.json'))
    copied += 1

print(f'Copied {copied} files to {dst}. Skipped {skipped}.')
