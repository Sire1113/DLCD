from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


TAXA_API_URL = "https://api.inaturalist.org/v1/taxa"
OBSERVATIONS_API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "dlcd/0.1 (educational image classification project)"
DEFAULT_CLASS_SPECS = ["dhole=dhole", "fox=red fox"]


@dataclass(frozen=True)
class ClassSpec:
    name: str
    query: str


@dataclass(frozen=True)
class SourceConfig:
    timeout: float = 90.0
    retries: int = 3
    backoff: float = 2.0


def parse_class_spec(value: str) -> ClassSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Class spec must use the form name=query")
    name, query = value.split("=", 1)
    name = name.strip()
    query = query.strip()
    if not name or not query:
        raise argparse.ArgumentTypeError("Class spec must include both name and query")
    return ClassSpec(name=name, query=query)


def fetch_json(url: str, params: dict[str, Any], *, timeout: float, retries: int, backoff: float) -> dict[str, Any]:
    query = urlencode(params)
    request = Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(backoff * (2**attempt))
    assert last_error is not None
    raise URLError(f"Failed to fetch {url} after {retries + 1} attempts: {last_error}") from last_error


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def search_taxon(query: str, *, timeout: float, retries: int, backoff: float) -> dict[str, Any]:
    payload = fetch_json(
        TAXA_API_URL,
        {
            "q": query,
            "rank": "species",
            "order": "desc",
            "order_by": "observations_count",
            "per_page": 10,
        },
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )
    results = payload.get("results", [])
    if not results:
        raise URLError(f"No iNaturalist taxon found for query: {query}")
    return results[0]


def fetch_observations(
    taxon_id: int,
    *,
    per_class: int,
    timeout: float,
    retries: int,
    backoff: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    page = 1
    while len(observations) < per_class:
        payload = fetch_json(
            OBSERVATIONS_API_URL,
            {
                "taxon_id": taxon_id,
                "photos": "true",
                "quality_grade": "research",
                "per_page": min(200, per_class),
                "page": page,
                "order": "desc",
                "order_by": "created_at",
            },
            timeout=timeout,
            retries=retries,
            backoff=backoff,
        )
        page_results = payload.get("results", [])
        if not page_results:
            break
        observations.extend(page_results)
        if len(page_results) < min(200, per_class):
            break
        page += 1
    return observations[:per_class]


def sanitize_filename(title: str, url: str) -> str:
    base_name = re.sub(r"[^\w.-]+", "_", title, flags=re.UNICODE).strip("._")
    suffix = Path(urlparse(url).path).suffix
    if not suffix:
        mime = mimetypes.guess_extension("image/jpeg") or ".jpg"
        suffix = mime
    if not base_name:
        base_name = sha256(url.encode("utf-8")).hexdigest()[:12]
    return f"{base_name}{suffix}"


def best_photo_url(photo: dict[str, Any]) -> str:
    url = str(photo.get("url", ""))
    if not url:
        return ""
    if "square." in url:
        return url.replace("square.", "large.")
    if "small." in url:
        return url.replace("small.", "large.")
    return url


def download_file(url: str, destination: Path, *, timeout: float, retries: int, backoff: float) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if destination.exists():
                destination.unlink(missing_ok=True)
            if attempt >= retries:
                break
            time.sleep(backoff * (2**attempt))
    assert last_error is not None
    raise URLError(f"Failed to download image after {retries + 1} attempts: {last_error}") from last_error


def download_class_images(
    output_dir: Path,
    class_spec: ClassSpec,
    per_class: int,
    *,
    timeout: float,
    retries: int,
    backoff: float,
) -> list[dict[str, Any]]:
    class_dir = output_dir / class_spec.name
    class_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    try:
        taxon = search_taxon(class_spec.query, timeout=timeout, retries=retries, backoff=backoff)
    except URLError as exc:
        return [
            {
                "class_name": class_spec.name,
                "query": class_spec.query,
                "status": "failed",
                "error": str(exc),
            }
        ]

    taxon_id = int(taxon["id"])
    observations = fetch_observations(
        taxon_id,
        per_class=per_class,
        timeout=timeout,
        retries=retries,
        backoff=backoff,
    )

    seen_urls: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for observation in observations:
        photos = observation.get("photos") or []
        if not photos:
            continue
        photo = photos[0]
        url = best_photo_url(photo)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        candidates.append(
            {
                "title": f"{taxon.get('name', class_spec.query)}_{observation.get('id', '')}",
                "url": url,
                "taxon": taxon,
                "observation_id": observation.get("id"),
                "photo_id": photo.get("id"),
                "license_code": photo.get("license_code", ""),
            }
        )
        if len(candidates) >= per_class:
            break

    for index, entry in enumerate(candidates, start=1):
        filename = f"{index:04d}_{sanitize_filename(entry['title'], entry['url'])}"
        destination = class_dir / filename
        try:
            download_file(entry["url"], destination, timeout=timeout, retries=retries, backoff=backoff)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            manifest.append(
                {
                    "class_name": class_spec.name,
                    "query": class_spec.query,
                    "title": entry["title"],
                    "url": entry["url"],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        manifest.append(
            {
                "class_name": class_spec.name,
                "query": class_spec.query,
                "title": entry["title"],
                "url": entry["url"],
                "path": str(destination),
                "status": "downloaded",
                "taxon_id": taxon_id,
                "taxon_name": taxon.get("name", ""),
                "taxon_preferred_common_name": taxon.get("preferred_common_name", ""),
                "observation_id": entry.get("observation_id"),
                "photo_id": entry.get("photo_id"),
                "license_code": entry.get("license_code", ""),
            }
        )

    if not manifest:
        manifest.append(
            {
                "class_name": class_spec.name,
                "query": class_spec.query,
                "status": "failed",
                "error": f"No downloadable photos found for iNaturalist taxon_id={taxon_id}",
            }
        )

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small binary image dataset from iNaturalist")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Target dataset directory")
    parser.add_argument("--per-class", type=int, default=120, help="Maximum downloaded images per class")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries per network request")
    parser.add_argument("--backoff", type=float, default=2.0, help="Exponential backoff base delay in seconds")
    parser.add_argument(
        "--class-spec",
        action="append",
        type=parse_class_spec,
        default=[],
        help="Class definition in the form name=query. May be repeated.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    class_specs = args.class_spec or [parse_class_spec(spec) for spec in DEFAULT_CLASS_SPECS]
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, Any]] = []
    for class_spec in class_specs:
        manifests.extend(
            download_class_images(
                output_dir,
                class_spec,
                args.per_class,
                timeout=args.timeout,
                retries=args.retries,
                backoff=args.backoff,
            )
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), "records": len(manifests)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
