from pathlib import Path
import csv
from typing import List, Dict, Optional, Set
import shutil

DEFAULT_HEADERS = [
    "title","abstract","doi","application_area","clinical_procedure","multimodal_imaging",
    "study_category","research_environment","purpose","imaging_technique","spectral_range"
]

class LabelCSVReader:
    def __init__(self, path: str, delimiter: str = ";", expected_headers: Optional[List[str]] = None):
        self.path = Path(path)
        self.delimiter = delimiter
        self.expected_headers = expected_headers or DEFAULT_HEADERS
        self._rows: List[Dict[str,str]] = []
        self._loaded = False

    def load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Label CSV not found: {self.path}")
        with self.path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            headers = [h.strip() for h in (reader.fieldnames or [])]
            missing = [h for h in self.expected_headers if h not in headers]
            if missing:
                raise ValueError(f"Missing expected headers: {missing}")
            self._rows = []
            for r in reader:
                # normalize keys: strip whitespace from header names and values
                norm = {k.strip(): (v.strip() if v is not None else "") for k,v in r.items()}
                self._rows.append(norm)
        self._loaded = True

    def rows(self) -> List[Dict[str,str]]:
        if not self._loaded:
            raise RuntimeError("Call load() before rows()")
        return self._rows

    def dois(self) -> Set[str]:
        if not self._loaded:
            raise RuntimeError("Call load() before dois()")
        s = set()
        for r in self._rows:
            doi = r.get("doi","").strip()
            if doi:
                s.add(doi)
        return s

    def unique_count(self) -> int:
        return len(self.dois())

    def sample_dois(self, n: int = 5) -> List[str]:
        s = list(self.dois())
        return s[:n]

    def lookup_by_doi(self) -> Dict[str, List[Dict[str,str]]]:
        if not self._loaded:
            raise RuntimeError("Call load() before lookup_by_doi()")
        d: Dict[str, List[Dict[str,str]]] = {}
        for r in self._rows:
            doi = r.get("doi","").strip()
            if not doi:
                continue
            d.setdefault(doi, []).append(r)
        return d

    def save_copy(self, out_dir: str, out_name: Optional[str] = None) -> str:
        """Copy original CSV to out_dir. Returns path of copied file."""
        out_dirp = Path(out_dir)
        out_dirp.mkdir(parents=True, exist_ok=True)
        out_name = out_name or self.path.name
        dst = out_dirp / out_name
        shutil.copy2(self.path, dst)
        return str(dst)