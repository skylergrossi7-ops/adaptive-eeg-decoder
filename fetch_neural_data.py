"""Fetch the prespecified 24 PhysioNet EEGMMIDB participants and verify each EDF."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import time
import mne

ROOT = Path(__file__).resolve().parent
FILES = [f"S{p:03}/S{p:03}R{r:02}.edf" for p in range(1, 25) for r in (4, 8, 12)]
BASE = "https://physionet.org/files/eegmmidb/1.0.0/"

def checksums():
    registry = Path(mne.__file__).parent / "data" / "eegbci_checksums.txt"
    return dict(line.split() for line in registry.read_text().splitlines() if line.strip())

def main():
    hashes = checksums()
    def fetch(name):
        dest = ROOT / "data" / "physionet" / name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == hashes[name]:
            return
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE + name, timeout=40) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest() != hashes[name]:
                    raise ValueError(f"Checksum mismatch: {name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for i, _ in enumerate(workers.map(fetch, FILES), 1):
            if i % 12 == 0:
                print(f"Verified {i}/{len(FILES)} files", flush=True)

if __name__ == "__main__":
    main()
