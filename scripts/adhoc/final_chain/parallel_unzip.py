import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

archive, target, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])


def extract(names):
    with zipfile.ZipFile(archive) as handle:
        for name in names:
            for attempt in range(3):
                try:
                    handle.extract(name, target)
                    break
                except FileExistsError:
                    continue
    return len(names)


started = time.time()
with zipfile.ZipFile(archive) as handle:
    infos = [info for info in handle.infolist() if not info.is_dir()]
infos.sort(key=lambda info: info.filename)
groups = [[info.filename for info in infos[index::workers * 8]] for index in range(workers * 8)]
done = 0
with ProcessPoolExecutor(workers) as pool:
    for count in pool.map(extract, groups):
        done += count
print(f"extracted {done} of {len(infos)} files in {time.time() - started:.0f} s", flush=True)
missing = [info.filename for info in infos if os.path.getsize(os.path.join(target, info.filename)) != info.file_size] if done == len(infos) else ["count"]
print("size mismatches", len(missing), missing[:3], flush=True)
sys.exit(0 if not missing else 1)
