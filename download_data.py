import os
import urllib.request
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "tmdb_5000_movies.csv": [
        "https://raw.githubusercontent.com/carmengcm/TMDB_5000_Movie_Dataset_PythonAnalysis/master/BaseDeDatos/tmdb_5000_movies.csv",
        "https://raw.githubusercontent.com/yc3404/tmdb5000/master/data/tmdb_5000_movies.csv",
        "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/main/tmdb_5000_movies.csv"
    ],
    "tmdb_5000_credits.csv": [
        "https://raw.githubusercontent.com/carmengcm/TMDB_5000_Movie_Dataset_PythonAnalysis/master/BaseDeDatos/tmdb_5000_credits.csv",
        "https://raw.githubusercontent.com/yc3404/tmdb5000/master/data/tmdb_5000_credits.csv"
    ]
}

def download_file(filename, urls):
    target = os.path.join(DATA_DIR, filename)
    if os.path.exists(target) and os.path.getsize(target) > 100000:
        print(f"{filename} already exists ({os.path.getsize(target):,} bytes). Skipping.")
        return True
    
    for url in urls:
        print(f"Downloading {filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response, open(target, 'wb') as out_file:
                chunk_size = 1024 * 1024
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    print(f"  Downloaded {downloaded:,} bytes...", end="\r")
            print(f"\nSuccessfully downloaded {filename} ({os.path.getsize(target):,} bytes)")
            return True
        except Exception as e:
            print(f"\nFailed from {url}: {e}")
            if os.path.exists(target):
                os.remove(target)
    return False

if __name__ == "__main__":
    success = True
    for fname, urls in FILES.items():
        if not download_file(fname, urls):
            print(f"ERROR: Could not download {fname}")
            success = False
    if not success:
        sys.exit(1)
    print("All dataset files ready in", DATA_DIR)
