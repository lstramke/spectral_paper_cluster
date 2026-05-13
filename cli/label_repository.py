from pathlib import Path
import csv
from typing import List, Dict, Optional
from document import Document, DocumentBatch
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

    def load(self) -> DocumentBatch:
        if not self.path.exists():
            raise FileNotFoundError(f"Label CSV not found: {self.path}")
        with self.path.open(encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj, delimiter=self.delimiter)
            headers = [header.strip() for header in (reader.fieldnames or [])]
            missing = [expected for expected in self.expected_headers if expected not in headers]
            if missing:
                raise ValueError(f"Missing expected headers: {missing}")
            rows: List[Dict[str, str]] = []
            for raw_row in reader:
                # normalize keys: strip whitespace from header names and values
                normalized = {key.strip(): (value.strip() if value is not None else "") for key, value in raw_row.items()}
                rows.append(normalized)

        documents: List[Document] = []
        for row in rows:
            title = row.get("title", "")
            abstract = row.get("abstract", "")
            doi = row.get("doi", "").strip()
            labels: Dict[str, str] = {}
            for header in headers:
                if header in ("title", "abstract", "doi"):
                    continue
                labels[header] = row.get(header, "")
            documents.append(Document(title=title, abstract=abstract, doi=doi, labels=labels))

        return DocumentBatch(headers=headers or self.expected_headers, documents=documents)

    def save_copy(self, out_dir: str, out_name: Optional[str] = None) -> str:
        """Copy original CSV to out_dir. Returns path of copied file."""
        out_dirp = Path(out_dir)
        out_dirp.mkdir(parents=True, exist_ok=True)
        out_name = out_name or self.path.name
        dst = out_dirp / out_name
        shutil.copy2(self.path, dst)
        return str(dst)
    
    def _write_rows(self, out_path: str, rows: List[Dict[str, str]], headers: Optional[List[str]] = None, delimiter: str = ";") -> str:
        """Write a list of dict rows to a CSV. Returns written file path."""
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = headers or self.expected_headers
        with outp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for r in rows:
                # ensure all expected keys exist
                row = {k: (r.get(k, "") if r.get(k, None) is not None else "") for k in fieldnames}
                writer.writerow(row)
        return str(outp)

    def save_final_csv(self, out_dir: str, rows: List[Dict[str, str]], out_name: Optional[str] = None, headers: Optional[List[str]] = None, delimiter: str = ";") -> str:
        """Convenience: write final labels CSV into out_dir and return path."""
        out_dirp = Path(out_dir)
        out_dirp.mkdir(parents=True, exist_ok=True)
        out_name = out_name or "final_labels.csv"
        return self._write_rows(str(out_dirp / out_name), rows, headers=headers, delimiter=delimiter)