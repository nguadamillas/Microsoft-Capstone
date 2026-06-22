import tarfile
import io
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, TED_BASE_URL, TED_PACKAGES


def download_package(package_id: str) -> Path:
    dest = RAW_DIR / f"{package_id}.tar.gz"
    if dest.exists():
        return dest

    url = f"{TED_BASE_URL}/{package_id}.tar.gz"
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as out_file:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                out_file.write(chunk)

    return dest


def extract_nested_archive(archive_path: Path, out_dir: Path) -> int:
    total_xml = 0
    with tarfile.open(archive_path, "r:gz") as outer:
        daily_packages = [m for m in outer.getmembers() if m.name.endswith(".tar.gz")]
        print(f"Found {len(daily_packages)} daily packages in {archive_path.name}")

        for daily in tqdm(daily_packages, desc=f"Processing {archive_path.name}"):
            f = outer.extractfile(daily)
            if f is None:
                continue
            data = io.BytesIO(f.read())

            with tarfile.open(fileobj=data, mode="r:gz") as inner:
                xml_members = [m for m in inner.getmembers() if m.name.endswith(".xml")]
                for member in xml_members:
                    member.name = Path(member.name).name
                    inner.extract(member, out_dir)
                    total_xml += 1
    return total_xml


RAW_DIR.mkdir(parents=True, exist_ok=True)
out_dir = RAW_DIR

existing_xml = list(RAW_DIR.glob("*.xml"))
if existing_xml:
    print(f"Found {len(existing_xml)} XML files in {RAW_DIR}. Skipping ingest.")
    sys.exit(0)

# If no archive is already present, download the configured TED packages.
archives = list(RAW_DIR.glob("*.tar.gz"))
if not archives:
    print("No archive files found in data/raw/. Downloading TED daily package archives...")
    for package_id in TED_PACKAGES:
        download_package(package_id)
    archives = list(RAW_DIR.glob("*.tar.gz"))

if not archives:
    raise FileNotFoundError(
        "No raw TED archives found in data/raw/ and automatic download failed. "
        "Please add raw package files or check your network connection.")

print("Extracting nested archives...")
total_xml = 0
for gz_path in archives:
    total_xml += extract_nested_archive(gz_path, out_dir)

print(f"Done: {total_xml:,} XML files extracted to data/raw/")
