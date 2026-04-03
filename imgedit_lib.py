"""
Helpers for ImgEdit (sysuyy/ImgEdit on Hugging Face) → WebDataset.

Schema is taken from the published parquets, not from other datasets:
- Most shards: input_images (list<string>), output_images (list<string>), prompt (string).
- Multiturn shards: column "data" = list<struct<input_images, output_images, prompt>>.

Paths in cells are repo-relative strings (often like results_remove/.../original.png).
After snapshot_download, they usually live under Singleturn/ or Multiturn/ once the
matching results_*.tar.split.* chunks are concatenated and extracted.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import tarfile
import unicodedata
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)

# --- Official ImgEdit parquet field names (HF dataset Parquet/*.parquet) ---
COL_INPUT_IMAGES = "input_images"
COL_OUTPUT_IMAGES = "output_images"
COL_PROMPT = "prompt"
COL_DATA_MULTITURN = "data"

# Fallback names if the project ever ships variants (path-string style)
PATH_ORIG_CANDIDATES: List[str] = [
    COL_INPUT_IMAGES,
    "original_path",
    "source_path",
    "input_path",
    "original",
]

PATH_EDIT_CANDIDATES: List[str] = [
    COL_OUTPUT_IMAGES,
    "edited_path",
    "target_path",
    "output_path",
    "edited",
    "result_path",
]

EMBED_ORIG_CANDIDATES: List[str] = ["source_img", "original_img", "input_img"]
EMBED_EDIT_CANDIDATES: List[str] = ["target_img", "edited_img", "output_img"]

TEXT_CANDIDATES: List[str] = [COL_PROMPT, "instruction", "caption", "text", "edit_instruction"]

# Parquet path strings often use "results_compose*" (e.g. results_compose_part0, results_compose_part6_fix;
# stage1_explore logs these as roots_missing_in_hub_filenames for hybrid_part*.parquet) while Hub ships
# results_hybrid* archives/trees. apply_rel_substitutions replaces the first occurrence of OLD in each path.
HYBRID_COMPOSE_SUBST: Tuple[str, str] = ("results_compose", "results_hybrid")

_TURN_ROOT_PREFIXES: Tuple[str, ...] = (
    "Singleturn/",
    "Multiturn/",
    "singleturn/",
    "multiturn/",
    "SINGLETURN/",
    "MULTITURN/",
)


def normalize_hub_relpath(s: str) -> str:
    """
    Normalize repo-relative paths from Hugging Face parquet cells / filenames for matching.

    NFC unicode, forward slashes, strip whitespace, drop redundant ``./`` and ``//``.
    """
    t = unicodedata.normalize("NFC", str(s).strip().replace("\\", "/"))
    while "//" in t:
        t = t.replace("//", "/")
    t = t.lstrip("/").lstrip("./")
    return t


def match_repo_paths(all_files: Sequence[str], pattern: str) -> List[str]:
    """
    Match Hugging Face ``list_repo_files`` paths against a shell-style glob.

    Uses normalized slashes / unicode first; if nothing matches, tries case-folding (some
    mirrors or older snapshots differ only by case); finally tries the raw pattern as given.
    """
    pat_norm = normalize_hub_relpath(pattern)

    def norm_key(path: str) -> str:
        return normalize_hub_relpath(path)

    out = sorted({f for f in all_files if fnmatch(norm_key(f), pat_norm)})
    if out:
        return out

    pat_cf = pat_norm.casefold()
    out = sorted({f for f in all_files if fnmatch(norm_key(f).casefold(), pat_cf)})
    if out:
        return out

    return sorted({f for f in all_files if fnmatch(f, pattern)})


def _parquet_rel_variants(raw: str) -> List[str]:
    """Deduped parquet path strings to try when resolving on disk (HF vs tar layout drift)."""
    n = normalize_hub_relpath(raw)
    out: List[str] = []
    seen: Set[str] = set()

    def add(x: str) -> None:
        if x and x not in seen:
            seen.add(x)
            out.append(x)

    add(n)
    add(raw.strip())
    cur = n
    for _ in range(4):
        stripped = False
        for pref in _TURN_ROOT_PREFIXES:
            pl, cl = pref.lower(), cur.lower()
            if cl.startswith(pl):
                cur = cur[len(pref) :].lstrip("/")
                add(cur)
                stripped = True
                break
        if not stripped or not cur:
            break
    return out


def _same_file_casefold(path: Path) -> Optional[Path]:
    """If ``path`` is missing, look for a same-parent file whose name matches case-insensitively."""
    if path.is_file():
        return path
    parent = path.parent
    if not parent.is_dir():
        return None
    want = path.name.casefold()
    try:
        for child in parent.iterdir():
            if child.is_file() and child.name.casefold() == want:
                return child
    except OSError:
        return None
    return None


def _resolve_under_any_results_task(
    dataset_root: Path, rel: Path, *, max_tasks: int = 4096
) -> Optional[Path]:
    """
    Last resort: parquet may name ``results_foo/part/file`` but Hub tar used a sibling
    ``results_bar``. Match on unique ``basename(parent)/filename`` under ``Singleturn/results_*``.
    """
    if len(rel.parts) < 2:
        return None
    parent_name, fname = rel.parts[-2], rel.parts[-1]
    single = dataset_root / "Singleturn"
    if not single.is_dir():
        return None
    hits: List[Path] = []
    try:
        scanned = 0
        for task in single.iterdir():
            if scanned >= max_tasks:
                break
            scanned += 1
            if not task.is_dir() or not task.name.startswith("results_"):
                continue
            c = task / parent_name / fname
            if c.is_file():
                hits.append(c)
            else:
                alt = _same_file_casefold(c)
                if alt is not None:
                    hits.append(alt)
    except OSError:
        return None
    if len(hits) == 1:
        return hits[0]
    return None


def apply_rel_substitutions(rel: str, rules: Sequence[Tuple[str, str]]) -> str:
    out = rel
    for old, new in rules:
        if old and new and old in out:
            out = out.replace(old, new, 1)
    return out


def apply_image_subdir(rel: str, subdir: str) -> str:
    """
    Prepend ``subdir`` to bare filenames (no directory component) before path resolution.

    Used for ``action_part*``-style parquets where cells are basename-only lists like
    ``['img.jpg']`` while images extract under e.g. ``Singleturn/part1/``.
    """
    if not (subdir or "").strip():
        return rel
    p = Path(str(rel).strip())
    if p.parent == Path("."):
        return str(Path(subdir.strip()) / p.name)
    return rel


def first_relpath_from_list_cell(cell: Any) -> Optional[str]:
    """
    ImgEdit stores one path as nested lists, e.g. [['results_remove/.../original.png']].
    Also accepts ['path'] or a bare string.
    """
    paths = flatten_path_strings(cell)
    return paths[0] if len(paths) == 1 else None


def flatten_path_strings(cell: Any) -> List[str]:
    """
    Collect all string paths from nested list structures (HF list columns).
    Used for strict checks: we only keep rows with exactly one input and one output path.
    """
    out: List[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
            return
        if isinstance(x, (list, tuple)):
            for item in x:
                walk(item)

    walk(cell)
    return out


def coerce_image_bytes(
    raw: object,
    *,
    parquet_path: Path,
    dataset_root: Path,
    subst_rules: Sequence[Tuple[str, str]] = (),
) -> Optional[bytes]:
    """HF 'image' feature as dict {bytes, path} or raw bytes."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        b = raw.get("bytes")
        if b is not None:
            return coerce_image_bytes(b, parquet_path=parquet_path, dataset_root=dataset_root, subst_rules=subst_rules)
        rel = raw.get("path")
        if rel:
            return path_string_to_bytes(
                str(rel),
                dataset_root=dataset_root,
                parquet_path=parquet_path,
                subst_rules=subst_rules,
            )
        return None
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return bytes(raw)
    return None


def resolve_dataset_image_path(
    path_str: str,
    *,
    dataset_root: Path,
    parquet_path: Path,
    subst_rules: Sequence[Tuple[str, str]] = (),
) -> Optional[Path]:
    """
    Resolve a relative image path from ImgEdit parquets.

    Tries dataset root, Singleturn/, Multiturn/, and the parquet file's directory
    (useful when Parquet/ lives next to extracted image trees).

    Also tries normalized / stripped ``Singleturn/`` prefixes from HF strings, case-insensitive
    filenames in the same parent directory, and a narrow scan under ``Singleturn/results_*``
    when the top-level ``results_*`` folder name in the parquet does not match the extracted tar.

    Absolute POSIX paths are used as-is if the file exists.
    """
    if path_str is None:
        return None
    s0 = apply_rel_substitutions(str(path_str).strip(), subst_rules)
    if not s0 or s0.lower() in ("none", "null"):
        return None

    p0 = Path(s0)
    if p0.is_absolute():
        hit = p0 if p0.is_file() else _same_file_casefold(p0)
        return hit

    variants = _parquet_rel_variants(s0)
    candidates: List[Path] = []
    for s in variants:
        p = Path(s)
        candidates.extend(
            [
                dataset_root / p,
                dataset_root / "Singleturn" / p,
                dataset_root / "Multiturn" / p,
                parquet_path.parent / p,
                dataset_root / s.lstrip("./"),
                dataset_root / "Singleturn" / s.lstrip("./"),
                dataset_root / "Multiturn" / s.lstrip("./"),
            ]
        )

    seen: Set[Path] = set()
    for c in candidates:
        try:
            r = c.resolve()
        except (OSError, RuntimeError):
            continue
        if r in seen:
            continue
        seen.add(r)
        if r.is_file():
            return r
        alt = _same_file_casefold(r)
        if alt is not None:
            try:
                ar = alt.resolve()
            except (OSError, RuntimeError):
                continue
            if ar not in seen:
                seen.add(ar)
                return ar

    for s in variants:
        tail = Path(s)
        if len(tail.parts) >= 2:
            hit = _resolve_under_any_results_task(dataset_root, tail)
            if hit is not None:
                return hit
    return None


def path_string_to_bytes(
    path_str: str,
    *,
    dataset_root: Path,
    parquet_path: Path,
    subst_rules: Sequence[Tuple[str, str]] = (),
) -> Optional[bytes]:
    p = resolve_dataset_image_path(path_str, dataset_root=dataset_root, parquet_path=parquet_path, subst_rules=subst_rules)
    if not p:
        return None
    try:
        return p.read_bytes()
    except OSError:
        return None


def encode_jpeg(
    image_bytes: bytes,
    *,
    max_side: Optional[int],
    jpeg_quality: int,
) -> Optional[bytes]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            if max_side and max(im.size) > max_side:
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality)
            return buf.getvalue()
    except Exception:
        return None


def add_tar_bytes(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def parquet_schema_columns(parquet_path: Path) -> List[str]:
    return list(pq.read_schema(parquet_path).names)


def detect_field_mapping(column_names: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Returns (orig_col, edit_col, text_col, mode); mode in
    {'imgedit_lists', 'imgedit_multiturn', 'embed', 'path', 'unknown'}.
    """
    lower = {c.lower(): c for c in column_names}

    def pick(cands: List[str]) -> Optional[str]:
        for c in cands:
            k = c.lower()
            if k in lower:
                return lower[k]
        return None

    if COL_DATA_MULTITURN in column_names:
        return None, None, COL_PROMPT, "imgedit_multiturn"

    if COL_INPUT_IMAGES in column_names and COL_OUTPUT_IMAGES in column_names:
        return COL_INPUT_IMAGES, COL_OUTPUT_IMAGES, pick(TEXT_CANDIDATES) or COL_PROMPT, "imgedit_lists"

    o_embed = pick(EMBED_ORIG_CANDIDATES)
    e_embed = pick(EMBED_EDIT_CANDIDATES)
    if o_embed and e_embed:
        return o_embed, e_embed, pick(TEXT_CANDIDATES), "embed"

    o_path = pick(PATH_ORIG_CANDIDATES)
    e_path = pick(PATH_EDIT_CANDIDATES)
    if o_path and e_path:
        return o_path, e_path, pick(TEXT_CANDIDATES), "path"

    if o_embed and e_path:
        return o_embed, e_path, pick(TEXT_CANDIDATES), "mixed_embed_orig"
    if o_path and e_embed:
        return o_path, e_embed, pick(TEXT_CANDIDATES), "mixed_embed_edit"

    return None, None, pick(TEXT_CANDIDATES), "unknown"


def iter_imgedit_pairs_from_row(
    row: Dict[str, Any],
    *,
    mode: str,
    orig_col: str,
    edit_col: str,
    text_col: Optional[str],
) -> Iterable[Tuple[Optional[str], Optional[str], str]]:
    """
    Yield (input_relpath, output_relpath, prompt) for training pairs.
    Skips multiturn turns with no image paths (instruction-only turns).
    """
    text_col = text_col or COL_PROMPT

    if mode == "imgedit_multiturn":
        turns = row.get(COL_DATA_MULTITURN) or []
        if not isinstance(turns, (list, tuple)):
            return
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            ins = flatten_path_strings(turn.get(COL_INPUT_IMAGES))
            outs = flatten_path_strings(turn.get(COL_OUTPUT_IMAGES))
            if len(ins) != 1 or len(outs) != 1:
                continue
            prompt = str(turn.get(text_col) or turn.get(COL_PROMPT) or "").strip()
            yield ins[0], outs[0], prompt
        return

    if mode == "imgedit_lists":
        # Top-level single slot per column (e.g. action_part*: one list per cell); flatten each
        # slot for nested [['path']] shapes. Skip rows with multiple top-level list elements.
        raw_in = row.get(orig_col)
        raw_out = row.get(edit_col)
        inputs = list(raw_in) if isinstance(raw_in, (list, tuple)) else ([] if raw_in is None else [raw_in])
        outputs = list(raw_out) if isinstance(raw_out, (list, tuple)) else ([] if raw_out is None else [raw_out])
        if len(inputs) == 1 and len(outputs) == 1:
            ins = flatten_path_strings(inputs[0])
            outs = flatten_path_strings(outputs[0])
            if len(ins) == 1 and len(outs) == 1:
                prompt = str(row.get(text_col) or "").strip()
                yield ins[0], outs[0], prompt
        return

    # path string mode: one path per column ( rare for ImgEdit HF)
    if mode == "path":
        inp = row.get(orig_col)
        out = row.get(edit_col)
        if isinstance(inp, (list, tuple)):
            ins = flatten_path_strings(inp)
            inp = ins[0] if len(ins) == 1 else None
        elif inp is not None:
            inp = str(inp).strip() or None
        if isinstance(out, (list, tuple)):
            outs = flatten_path_strings(out)
            out = outs[0] if len(outs) == 1 else None
        elif out is not None:
            out = str(out).strip() or None
        prompt = str(row.get(text_col) or "").strip()
        if inp and out:
            yield str(inp), str(out), prompt


def read_first_row_dict(parquet_path: Path) -> Optional[Dict[str, Any]]:
    try:
        t = pq.read_table(parquet_path, columns=parquet_schema_columns(parquet_path))
    except Exception:
        return None
    if t.num_rows == 0:
        return None
    batch = t.slice(0, 1).to_pydict()
    return {k: batch[k][0] for k in batch}


def sanity_check_tar_triplet(
    tar_path: str,
    expected_samples: int,
    *,
    decode_samples: int,
    left_ext: str = "jpg",
    right_ext: str = "jpg2",
    text_ext: str = "txt",
) -> Dict[str, int]:
    """Each sample key must have .{left_ext}, .{right_ext}, .{text_ext} (WebDataset-style)."""
    with tarfile.open(tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        by_key: DefaultDict[str, Set[str]] = defaultdict(set)
        counts = {left_ext: 0, right_ext: 0, text_ext: 0}

        for m in members:
            if "." not in m.name:
                continue
            base, suffix = m.name.rsplit(".", 1)
            if suffix in counts:
                by_key[base].add(suffix)
                counts[suffix] += 1

        required = {left_ext, right_ext, text_ext}
        bad_keys = [k for k, got in by_key.items() if got != required]
        if bad_keys:
            raise RuntimeError(
                f"Sanity failed: {len(bad_keys)} keys missing required suffixes "
                f"{required}. Example: {bad_keys[:3]}"
            )

        if not (counts[left_ext] == counts[right_ext] == counts[text_ext] == expected_samples):
            raise RuntimeError(
                f"Sanity failed: expected {expected_samples} samples, got "
                f"{left_ext}={counts[left_ext]} {right_ext}={counts[right_ext]} "
                f"{text_ext}={counts[text_ext]}"
            )

        sample_keys = sorted(by_key.keys())[: max(0, decode_samples)]
        for key in sample_keys:
            a = tf.extractfile(f"{key}.{left_ext}").read()
            b = tf.extractfile(f"{key}.{right_ext}").read()
            txt_bytes = tf.extractfile(f"{key}.{text_ext}").read()
            with Image.open(io.BytesIO(a)) as im:
                im.verify()
            with Image.open(io.BytesIO(b)) as im2:
                im2.verify()
            txt_bytes.decode("utf-8")

    return dict(counts)


def warn_bucket_zone(bucket: str, expected_zone: str) -> None:
    """
    If expected_zone is set, warn when it does not appear in the bucket URI.
    Same guardrail as deepfusion/dataset/upload_magicbrush_webdataset.py.
    """
    b = (bucket or "").strip()
    ez = (expected_zone or "").strip()
    if not ez or not b:
        return
    if ez not in b:
        logger.warning(
            "Expected zone marker %r not found in bucket %r. "
            "Match your VM region/zone to the bucket to avoid slow uploads and egress.",
            ez,
            b,
        )


def dir_size_human(path: Path) -> str:
    ret = subprocess.run(
        ["du", "-sh", str(path)],
        capture_output=True,
        text=True,
    )
    if ret.returncode != 0 or not ret.stdout.strip():
        return "N/A"
    return ret.stdout.strip().split()[0]


def log_paths_disk_usage(label: str, paths: Sequence[Path]) -> None:
    """Log mount space and `du -sh` for paths (deepfusion MagicBrush uploader style)."""
    plist = [p for p in paths if p is not None]
    if not plist:
        return
    try:
        usage = shutil.disk_usage(plist[0])
        gb = 1024**3
        msg = (
            f"{label}: free={usage.free / gb:.2f}GiB used={usage.used / gb:.2f}GiB "
            f"total={usage.total / gb:.2f}GiB"
        )
    except OSError as exc:
        logger.info("%s: disk usage unavailable (%s)", label, exc)
        return
    parts = [msg]
    for p in plist:
        if p.exists():
            parts.append(f"{p}={dir_size_human(p)}")
        else:
            parts.append(f"{p}=missing")
    logger.info(" | ".join(parts))


def gcs_object_exists(gcs_uri: str) -> bool:
    ret = subprocess.run(
        ["gcloud", "storage", "ls", gcs_uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return ret.returncode == 0


def upload_shard_noclobber(local_path: str, bucket_prefix: str) -> None:
    dest = f"{bucket_prefix.rstrip('/')}/"
    subprocess.run(
        ["gcloud", "storage", "cp", "-n", local_path, dest],
        check=True,
    )


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def summarize_parquet(parquet_path: Path, max_types: int = 16) -> Dict[str, Any]:
    schema = pq.read_schema(parquet_path)
    cols = list(schema.names)
    meta = pq.read_metadata(parquet_path)
    preview_types: Dict[str, str] = {}
    for name in cols[:max_types]:
        try:
            preview_types[name] = str(schema.field(name).type)
        except Exception:
            preview_types[name] = "?"
    mapping = detect_field_mapping(cols)
    return {
        "path": str(parquet_path),
        "num_rows": meta.num_rows,
        "num_columns": len(cols),
        "columns": cols,
        "column_types_preview": preview_types,
        "detected_mode": mapping[3],
        "detected_columns": {"original": mapping[0], "edited": mapping[1], "text": mapping[2]},
    }
