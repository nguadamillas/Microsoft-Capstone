import tarfile, io, sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

gz_path = RAW_DIR / "2026-1.tar.gz"
out_dir = RAW_DIR

print("Extracting nested archives...")
total_xml = 0

with tarfile.open(gz_path, "r:gz") as outer:
    daily_packages = [m for m in outer.getmembers() if m.name.endswith(".tar.gz")]
    print(f"Found {len(daily_packages)} daily packages")
    
    for daily in tqdm(daily_packages, desc="Processing daily packages"):
        # Extract the inner tar.gz into memory
        f = outer.extractfile(daily)
        if f is None:
            continue
        data = io.BytesIO(f.read())
        
        # Open the inner archive and extract XML files
        with tarfile.open(fileobj=data, mode="r:gz") as inner:
            xml_members = [m for m in inner.getmembers() if m.name.endswith(".xml")]
            for member in xml_members:
                member.name = Path(member.name).name  # flatten path
                inner.extract(member, out_dir)
                total_xml += 1

print(f"Done: {total_xml:,} XML files extracted to data/raw/")
