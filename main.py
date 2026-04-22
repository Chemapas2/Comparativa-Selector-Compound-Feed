from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Comparador de fórmulas | Rebranding",
    page_icon="📊",
    layout="wide",
)

NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:\.\d+)?$|^\.$")
SPEC_LINE_RE = re.compile(
    r"Specification:\s*([^\s]+)\s+(.+?)\s*:\s*Cost/tonne:\s*([0-9.]+)",
    re.IGNORECASE,
)
SINGLE_MIX_SPEC_LINE_RE = re.compile(
    r"SP:\s*([^\s]+)\s+(.+?)\s+[0-9.,]+\s*%\s*,\s*[0-9.,]+\s*Kg\s*\(Recost:\s*([0-9.,]+)\)",
    re.IGNORECASE,
)

SPECIES_RULES: List[Tuple[str, List[str]]] = [
    (
        "Rumiantes",
        [
            "ovi",
            "ovino",
            "capri",
            "capr",
            "rumi",
            "corder",
            "corde",
            "lactocorder",
            "cebo int",
            "cabrito",
            "ovigen",
            "vacuno",
            "novilla",
            "dairy",
            "ternero",
        ],
    ),
    (
        "Porcino",
        [
            "porc",
            "lech",
            "cerd",
            "iber",
            "optiporc",
            "baby",
            "cochin",
            "prestarter",
            "starter porc",
        ],
    ),
    (
        "Avicultura",
        [
            "avi",
            "poll",
            "broiler",
            "poned",
            "puesta",
            "recria",
            "recría",
            "pavo",
            "perd",
            "gallina",
            "campera",
            "erliva",
            "biofeed",
        ],
    ),
    ("Caballos", ["caball", "equi", "horse"]),
]

DEFAULT_COMPONENT_WEIGHTS = {
    "nutrients": 0.50,
    "ingredients": 0.25,
    "limits": 0.15,
    "price": 0.10,
}


@dataclass
class Repository:
    filename: str
    products: pd.DataFrame
    analyses: pd.DataFrame
    ingredients: pd.DataFrame
    nutrient_map: Dict[str, Dict[str, float]]
    ingredient_pct_map: Dict[str, Dict[str, float]]
    ingredient_detail_map: Dict[str, Dict[str, dict]]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_is_numeric(token: str) -> bool:
    return bool(NUMERIC_TOKEN_RE.match(str(token).strip()))


def safe_float(value) -> float:
    if value is None:
        return np.nan
    value = str(value).strip()
    if value in {"", ".", "-", "nan", "None"}:
        return np.nan
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return np.nan


def display_float(value: float, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def species_pref_key(species: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", normalize_text(species).lower()).strip("_")
    return key or "global"


def load_readme_text() -> str:
    candidates = [
        Path(__file__).with_name("README.md"),
        Path.cwd() / "README.md",
        Path("/mnt/data/README.md"),
    ]
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        except Exception:
            continue
    return "README.md no disponible en esta ejecución."


def heuristic_help_markdown() -> str:
    return """
### Cómo interpreta la app la similitud entre piensos

La aplicación no decide automáticamente si un producto destino es válido o no. Construye un **ranking heurístico** para ayudarte a priorizar qué referencias estándar merecen revisión técnica.

**Qué entra en el score total**
- **Nutrientes**: compara las analíticas seleccionadas por el usuario.
- **Ingredientes**: compara porcentajes de materias primas.
- **Límites**: penaliza incompatibilidades de mínimos, máximos o tipo de restricción.
- **Precio**: incorpora la desviación económica respecto al pienso origen.

**Cómo leer el resultado**
- **Score más bajo = mayor parecido relativo** dentro de esa comparación.
- El score sirve para **ordenar candidatos**, no para aprobar automáticamente un cambio.
- Un candidato puede quedar bien posicionado de forma global y aun así requerir revisión si falla en una métrica crítica o en un límite de ingrediente.

**Qué conviene revisar siempre antes de decidir**
1. Nutrientes críticos de la especie.
2. Materias primas diferenciales y sus límites.
3. Diferencia de precio.
4. Encaje técnico-comercial y comportamiento esperado en campo.

**Uso recomendado**
Utiliza el ranking para reducir de forma drástica la revisión manual y valida técnicamente los 3-5 primeros candidatos.
"""


def get_metric_preferences(species: str, common_metrics: Iterable[str]) -> Tuple[List[str], Dict[str, float]]:
    metrics_available = list(common_metrics)
    prefs = st.session_state.setdefault("metric_preferences", {})
    species_key = species_pref_key(species)
    species_prefs = prefs.get(species_key, {})

    saved_metrics = [metric for metric in species_prefs.get("selected_metrics", []) if metric in metrics_available]
    if not saved_metrics:
        saved_metrics = auto_select_metrics(metrics_available, species)

    saved_weights = species_prefs.get("metric_weights", {})
    resolved_weights = {
        metric: float(saved_weights.get(metric, default_metric_weight(metric, species)))
        for metric in metrics_available
    }
    return saved_metrics, resolved_weights


def save_metric_preferences(species: str, selected_metrics: List[str], metric_weights: Dict[str, float]) -> None:
    prefs = st.session_state.setdefault("metric_preferences", {})
    species_key = species_pref_key(species)
    existing = prefs.get(species_key, {})
    merged_weights = dict(existing.get("metric_weights", {}))
    merged_weights.update({metric: float(weight) for metric, weight in metric_weights.items()})
    prefs[species_key] = {
        "selected_metrics": list(selected_metrics),
        "metric_weights": merged_weights,
    }


def reset_metric_preferences(species: str) -> None:
    species_key = species_pref_key(species)
    prefs = st.session_state.setdefault("metric_preferences", {})
    prefs.pop(species_key, None)
    multiselect_key = f"selected_metrics_{species_key}"
    if multiselect_key in st.session_state:
        del st.session_state[multiselect_key]


def stable_metrics_signature(metrics: Iterable[str]) -> str:
    joined = "|".join(metrics)
    return md5(joined.encode("utf-8")).hexdigest()[:10]


def infer_species(product_name: str, filename: str = "") -> str:
    text = f"{filename} {product_name}".lower()
    for label, patterns in SPECIES_RULES:
        if any(pattern in text for pattern in patterns):
            return label
    return "Sin clasificar"


@st.cache_data(show_spinner=False)
def read_lines_from_excel(file_bytes: bytes, filename: str) -> Tuple[List[str], str]:
    """Read an uploaded workbook and reconstruct one logical text line per row.

    The sample repository file is a print/export where each logical line may live in one
    or more cells. Joining the non-empty cells row-wise makes the parser robust to both
    one-column and multi-column exports.
    """
    workbook = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, dtype=str)
    if not workbook:
        raise ValueError(f"No se ha podido leer ninguna hoja en {filename}.")

    first_sheet_name = list(workbook.keys())[0]
    sheet = workbook[first_sheet_name].fillna("")
    lines: List[str] = []
    for _, row in sheet.iterrows():
        parts = [normalize_text(cell) for cell in row.tolist() if normalize_text(cell)]
        if parts:
            lines.append(" ".join(parts))
        else:
            lines.append("")
    return lines, first_sheet_name


def is_formula_start_line(line: str) -> bool:
    normalized = normalize_text(line)
    return "Specification:" in normalized or bool(SINGLE_MIX_SPEC_LINE_RE.search(normalized))


def split_formula_blocks(lines: List[str]) -> List[List[str]]:
    starts = [idx for idx, line in enumerate(lines) if is_formula_start_line(line)]
    if not starts:
        raise ValueError(
            "No se han encontrado bloques reconocibles de fórmula. La app espera exportaciones tipo Multi-Mix ('Specification:') o Single-Mix ('SP:')."
        )

    blocks: List[List[str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines) - 1
        block = lines[start : end + 1]
        blocks.append(block)
    return blocks


def parse_spec_line(line: str) -> dict:
    normalized = normalize_text(line)

    match = SPEC_LINE_RE.search(normalized)
    if match:
        raw_product = normalize_text(match.group(2))
        code, product_name = split_product_code_and_name(raw_product)
        return {
            "spec_id": match.group(1),
            "product_raw": raw_product,
            "product_code": code,
            "product_name": product_name,
            "cost_tonne": safe_float(match.group(3)),
            "source_format": "multimix",
        }

    match = SINGLE_MIX_SPEC_LINE_RE.search(normalized)
    if match:
        raw_product = normalize_text(match.group(2))
        return {
            "spec_id": match.group(1),
            "product_raw": raw_product,
            "product_code": "",
            "product_name": raw_product,
            "cost_tonne": safe_float(match.group(3)),
            "source_format": "singlemix",
        }

    return {}


def split_product_code_and_name(raw_product: str) -> Tuple[str, str]:
    raw_product = normalize_text(raw_product)
    if "." in raw_product:
        code, rest = raw_product.split(".", 1)
        return normalize_text(code), normalize_text(rest)
    return "", raw_product


def find_section_indices(block_lines: List[str]) -> dict:
    positions = {
        "ingredients_start": None,
        "analysis_start": None,
        "rejected_start": None,
        "sensitivity_start": None,
    }
    for idx, line in enumerate(block_lines):
        normalized = normalize_text(line).upper()
        if "INCLUDED RAW MATERIALS" in normalized:
            positions["ingredients_start"] = idx + 2
        elif "REJECTED RAW MATERIALS" in normalized and positions["rejected_start"] is None:
            positions["rejected_start"] = idx
        elif normalized.startswith("ANALYSIS") or "NUTRIENT ANALYSIS" in normalized:
            positions["analysis_start"] = idx + 2
        elif "RAW MATERIAL SENSITIVITY" in normalized:
            positions["sensitivity_start"] = idx
            break
    return positions


def parse_ingredient_line_multimix(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 7:
        return None

    has_limit_token = not token_is_numeric(tokens[-3])
    max_token = tokens[-1]
    min_token = tokens[-2]

    if has_limit_token:
        limit_token = tokens[-3].upper()
        tonnes_token = tokens[-4]
        kilos_token = tokens[-5]
        pct_token = tokens[-6]
        avg_cost_token = tokens[-7]
        head = tokens[:-7]
    else:
        limit_token = ""
        tonnes_token = tokens[-3]
        kilos_token = tokens[-4]
        pct_token = tokens[-5]
        avg_cost_token = tokens[-6]
        head = tokens[:-6]

    if len(head) < 2:
        return None

    ingredient_key = normalize_text(head[0])
    ingredient_name = normalize_text(" ".join(head[1:]))
    ingredient_label = normalize_text(f"{ingredient_key} {ingredient_name}")

    return {
        "ingredient_key": ingredient_key,
        "ingredient_name": ingredient_name,
        "ingredient_label": ingredient_label,
        "avg_cost": safe_float(avg_cost_token),
        "pct": safe_float(pct_token),
        "kilos": safe_float(kilos_token),
        "tonnes": safe_float(tonnes_token),
        "limit_type": limit_token,
        "min": safe_float(min_token),
        "max": safe_float(max_token),
    }


def parse_ingredient_line_singlemix(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 8:
        return None

    start_idx = None
    for idx in range(1, len(tokens) - 2):
        if token_is_numeric(tokens[idx]) and token_is_numeric(tokens[idx + 1]) and token_is_numeric(tokens[idx + 2]):
            pct_value = safe_float(tokens[idx])
            kilos_value = safe_float(tokens[idx + 1])
            avg_cost_value = safe_float(tokens[idx + 2])
            pct_ok = pd.isna(pct_value) or (0.0 <= pct_value <= 100.0)
            kilos_ok = pd.notna(kilos_value) and (pd.isna(pct_value) or kilos_value >= pct_value)
            avg_cost_ok = pd.notna(avg_cost_value) and avg_cost_value >= 10.0
            if pct_ok and kilos_ok and avg_cost_ok:
                start_idx = idx
                break

    if start_idx is None or start_idx < 2:
        return None

    head = tokens[:start_idx]
    ingredient_key = normalize_text(head[0])
    ingredient_name = normalize_text(" ".join(head[1:]))
    ingredient_label = normalize_text(f"{ingredient_key} {ingredient_name}")

    pct_token = tokens[start_idx]
    kilos_token = tokens[start_idx + 1]
    avg_cost_token = tokens[start_idx + 2]
    idx = start_idx + 3

    limit_token = ""
    if idx < len(tokens) and not token_is_numeric(tokens[idx]):
        limit_token = normalize_text(tokens[idx]).upper()
        idx += 1

    min_token = tokens[idx] if idx < len(tokens) else np.nan
    max_token = tokens[idx + 1] if idx + 1 < len(tokens) else np.nan
    kilos_value = safe_float(kilos_token)

    return {
        "ingredient_key": ingredient_key,
        "ingredient_name": ingredient_name,
        "ingredient_label": ingredient_label,
        "avg_cost": safe_float(avg_cost_token),
        "pct": safe_float(pct_token),
        "kilos": kilos_value,
        "tonnes": kilos_value / 1000.0 if pd.notna(kilos_value) else np.nan,
        "limit_type": limit_token,
        "min": safe_float(min_token),
        "max": safe_float(max_token),
    }


def parse_ingredient_line(line: str, source_format: str) -> dict | None:
    if source_format == "singlemix":
        return parse_ingredient_line_singlemix(line)
    return parse_ingredient_line_multimix(line)


def parse_analysis_line_multimix(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 2:
        return None

    if tokens[:3] == ["[", "PESO", "]"]:
        analysis_name = "[PESO]"
        rest = tokens[3:]
    else:
        analysis_name = tokens[0]
        rest = tokens[1:]

    if not rest:
        return None

    level = safe_float(rest[0])
    idx = 1
    limit_type = ""
    if idx < len(rest) and not token_is_numeric(rest[idx]):
        limit_type = rest[idx].upper()
        idx += 1

    minimum = safe_float(rest[idx]) if idx < len(rest) else np.nan
    maximum = safe_float(rest[idx + 1]) if idx + 1 < len(rest) else np.nan
    without_dummies = safe_float(rest[-1]) if rest else np.nan

    return {
        "analysis_name": normalize_text(analysis_name),
        "level": level,
        "limit_type": limit_type,
        "min": minimum,
        "max": maximum,
        "without_dummies": without_dummies,
    }


def parse_analysis_line_singlemix(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 3:
        return None

    if tokens[:3] == ["[", "PESO", "]"]:
        analysis_name = "[PESO]"
        rest = tokens[3:]
    else:
        analysis_name = tokens[0]
        rest = tokens[1:]
        if rest and not token_is_numeric(rest[0]):
            rest = rest[1:]

    if not rest:
        return None

    level = safe_float(rest[0])
    idx = 1
    limit_type = ""
    if idx < len(rest) and not token_is_numeric(rest[idx]):
        limit_type = normalize_text(rest[idx]).upper()
        idx += 1

    minimum = safe_float(rest[idx]) if idx < len(rest) else np.nan
    maximum = safe_float(rest[idx + 1]) if idx + 1 < len(rest) else np.nan

    return {
        "analysis_name": normalize_text(analysis_name),
        "level": level,
        "limit_type": limit_type,
        "min": minimum,
        "max": maximum,
        "without_dummies": np.nan,
    }


def parse_analysis_line(line: str, source_format: str) -> dict | None:
    if source_format == "singlemix":
        return parse_analysis_line_singlemix(line)
    return parse_analysis_line_multimix(line)


@st.cache_data(show_spinner=False)
def parse_repository(file_bytes: bytes, filename: str) -> Repository:
    lines, sheet_name = read_lines_from_excel(file_bytes, filename)
    blocks = split_formula_blocks(lines)

    product_rows: List[dict] = []
    analysis_rows: List[dict] = []
    ingredient_rows: List[dict] = []

    for block in blocks:
        spec_meta = parse_spec_line(block[0])
        if not spec_meta:
            continue

        source_format = spec_meta.get("source_format", "multimix")
        positions = find_section_indices(block)
        product_id = f"{spec_meta['spec_id']}__{spec_meta['product_raw']}"
        species = infer_species(spec_meta["product_name"], filename)

        product_rows.append(
            {
                "product_id": product_id,
                "spec_id": spec_meta["spec_id"],
                "product_code": spec_meta["product_code"],
                "product_raw": spec_meta["product_raw"],
                "product_name": spec_meta["product_name"],
                "display_name": f"{spec_meta['product_name']} ({spec_meta['spec_id']})",
                "species": species,
                "cost_tonne": spec_meta["cost_tonne"],
                "source_file": filename,
                "sheet_name": sheet_name,
                "source_format": source_format,
            }
        )

        if positions["ingredients_start"] is not None and positions["analysis_start"] is not None:
            ingredient_end = positions["analysis_start"] - 2
            if positions.get("rejected_start") is not None:
                ingredient_end = min(ingredient_end, positions["rejected_start"])
            if source_format == "singlemix":
                ingredient_end = positions["analysis_start"] - 1
                if positions.get("rejected_start") is not None:
                    ingredient_end = min(ingredient_end, positions["rejected_start"])
            ingredient_lines = block[positions["ingredients_start"] : max(positions["ingredients_start"], ingredient_end)]
            for line in ingredient_lines:
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.startswith("-")
                    or "REJECTED RAW MATERIALS" in stripped.upper()
                    or "NUTRIENT ANALYSIS" in stripped.upper()
                ):
                    continue
                parsed = parse_ingredient_line(line, source_format=source_format)
                if parsed:
                    ingredient_rows.append({"product_id": product_id, **parsed})

        if positions["analysis_start"] is not None:
            end = positions["sensitivity_start"] if positions["sensitivity_start"] is not None else len(block)
            analysis_lines = block[positions["analysis_start"] : end]
            for line in analysis_lines:
                stripped = line.strip()
                upper = stripped.upper()
                if (
                    not stripped
                    or stripped.startswith("-")
                    or stripped.startswith(":")
                    or stripped.startswith("=")
                    or upper.startswith("_X")
                    or "RAW MATERIAL SENSITIVITY" in upper
                    or "REJECTED RAW MATERIALS" in upper
                    or "INCLUDED RAW MATERIALS" in upper
                ):
                    continue
                parsed = parse_analysis_line(line, source_format=source_format)
                if parsed:
                    analysis_rows.append({"product_id": product_id, **parsed})

    products_df = pd.DataFrame(product_rows)
    analyses_df = pd.DataFrame(analysis_rows)
    ingredients_df = pd.DataFrame(ingredient_rows)

    if products_df.empty:
        raise ValueError(f"No se ha podido extraer ningún producto del archivo {filename}.")

    if analyses_df.empty:
        analyses_df = pd.DataFrame(
            columns=["product_id", "analysis_name", "level", "limit_type", "min", "max", "without_dummies"]
        )
    if ingredients_df.empty:
        ingredients_df = pd.DataFrame(
            columns=[
                "product_id",
                "ingredient_key",
                "ingredient_name",
                "ingredient_label",
                "avg_cost",
                "pct",
                "kilos",
                "tonnes",
                "limit_type",
                "min",
                "max",
            ]
        )

    products_df = products_df.sort_values(["species", "product_name", "spec_id"]).reset_index(drop=True)
    analyses_df = analyses_df.sort_values(["product_id", "analysis_name"]).reset_index(drop=True)
    ingredients_df = ingredients_df.sort_values(["product_id", "ingredient_label"]).reset_index(drop=True)

    nutrient_map: Dict[str, Dict[str, float]] = {
        product_id: group.set_index("analysis_name")["level"].to_dict()
        for product_id, group in analyses_df.groupby("product_id")
    }
    ingredient_pct_map: Dict[str, Dict[str, float]] = {
        product_id: group.set_index("ingredient_label")["pct"].to_dict()
        for product_id, group in ingredients_df.groupby("product_id")
    }
    ingredient_detail_map: Dict[str, Dict[str, dict]] = {}
    for product_id, group in ingredients_df.groupby("product_id"):
        details = {}
        for _, row in group.iterrows():
            details[row["ingredient_label"]] = {
                "pct": safe_float(row["pct"]),
                "limit_type": normalize_text(row["limit_type"]).upper(),
                "min": safe_float(row["min"]),
                "max": safe_float(row["max"]),
                "avg_cost": safe_float(row["avg_cost"]),
            }
        ingredient_detail_map[product_id] = details

    return Repository(
        filename=filename,
        products=products_df,
        analyses=analyses_df,
        ingredients=ingredients_df,
        nutrient_map=nutrient_map,
        ingredient_pct_map=ingredient_pct_map,
        ingredient_detail_map=ingredient_detail_map,
    )


def metric_group(metric_name: str) -> str:
    m = metric_name.upper()
    if any(token in m for token in ["P_D_I_", "PDI", "PROT", "PB", "CP", "MAT_PROT"]):
        return "protein"
    if any(token in m for token in ["LYS", "LIS", "MET", "TRE", "THR", "TRP", "VAL", "ILE", "ARG", "CYS"]):
        return "amino"
    if any(token in m for token in ["U_F_", "UFL", "UFC", "ENER", "EM", "ENL", "ME", "NEL", "NE ", "E_L_"]):
        return "energy"
    if any(token in m for token in ["CA", "P_", "PHOS", "MG", "NA", "CL", "K_"]):
        return "minerals"
    if any(token in m for token in ["VIT", "COLINA", "BIOT", "OHD"]):
        return "vitamins"
    return "other"


def default_metric_weight(metric_name: str, species: str) -> float:
    group = metric_group(metric_name)
    species = species.lower()

    if "rumi" in species:
        mapping = {
            "energy": 2.5,
            "protein": 2.8,
            "amino": 1.6,
            "minerals": 1.8,
            "vitamins": 1.2,
            "other": 1.0,
        }
        if metric_name.upper() in {"P_D_I_E", "P_D_I_N"}:
            return 3.5
    else:
        mapping = {
            "energy": 2.4,
            "protein": 2.4,
            "amino": 3.0,
            "minerals": 1.8,
            "vitamins": 1.2,
            "other": 1.0,
        }
        if metric_name.upper() in {"LYS", "LIS", "MET", "TRE", "THR", "TRP"}:
            return 3.2

    if metric_name.upper() in {"CA", "P_", "MG"}:
        return max(mapping[group], 2.0)
    return mapping[group]


def auto_select_metrics(available_metrics: Iterable[str], species: str) -> List[str]:
    metrics = list(available_metrics)
    if not metrics:
        return []

    priority_patterns: List[re.Pattern] = []
    if species.lower() == "rumiantes":
        priority_patterns = [
            re.compile(r"U_F_|UFL|UFC|ENER|EM|ENL|E_L_", re.I),
            re.compile(r"PROT|P_D_I_E|P_D_I_N|PDI|PB|CP", re.I),
            re.compile(r"CA|P_|MG|NA|CL", re.I),
            re.compile(r"VIT|COLINA|OHD", re.I),
        ]
    else:
        priority_patterns = [
            re.compile(r"ENER|EM|ENL|ME|NEL|E_L_|U_F_", re.I),
            re.compile(r"PROT|PB|CP", re.I),
            re.compile(r"LYS|LIS|MET|TRE|THR|TRP|VAL|ILE", re.I),
            re.compile(r"CA|P_|MG|NA|CL", re.I),
            re.compile(r"VIT|COLINA|OHD", re.I),
        ]

    selected: List[str] = []
    for pattern in priority_patterns:
        selected.extend([metric for metric in metrics if pattern.search(metric) and metric not in selected])

    if len(selected) < 10:
        selected.extend([metric for metric in metrics if metric not in selected])

    return selected[: min(12, len(selected))]


def build_scale_map(
    base_product_id: str,
    candidate_ids: List[str],
    source_repo: Repository,
    target_repo: Repository,
    selected_metrics: List[str],
) -> Dict[str, float]:
    scale_map: Dict[str, float] = {}
    for metric in selected_metrics:
        values = []
        base_value = source_repo.nutrient_map.get(base_product_id, {}).get(metric)
        if pd.notna(base_value):
            values.append(base_value)
        for product_id in candidate_ids:
            value = target_repo.nutrient_map.get(product_id, {}).get(metric)
            if pd.notna(value):
                values.append(value)
        if len(values) <= 1:
            scale = max(abs(values[0]), 1.0) if values else 1.0
        else:
            scale = float(np.nanstd(values, ddof=0))
            if not np.isfinite(scale) or scale < 1e-6:
                scale = max(abs(np.nanmedian(values)), 1.0)
        scale_map[metric] = scale
    return scale_map


def ingredient_union(
    base_ingredients: Dict[str, dict],
    candidate_ingredients: Dict[str, dict],
    threshold_pct: float = 0.25,
) -> List[str]:
    names = set()
    for ingredient_name, detail in base_ingredients.items():
        if (detail.get("pct") or 0) >= threshold_pct or any(pd.notna(detail.get(k)) for k in ["min", "max"]):
            names.add(ingredient_name)
    for ingredient_name, detail in candidate_ingredients.items():
        if (detail.get("pct") or 0) >= threshold_pct or any(pd.notna(detail.get(k)) for k in ["min", "max"]):
            names.add(ingredient_name)
    return sorted(names)


def compute_limit_penalty(base_detail: dict, candidate_detail: dict) -> float:
    penalty = 0.0
    base_pct = base_detail.get("pct", 0.0) or 0.0
    candidate_pct = candidate_detail.get("pct", 0.0) or 0.0

    for ref_detail, other_pct in [(base_detail, candidate_pct), (candidate_detail, base_pct)]:
        minimum = ref_detail.get("min")
        maximum = ref_detail.get("max")
        if pd.notna(minimum) and other_pct < minimum:
            penalty += (minimum - other_pct) / max(minimum, 1.0)
        if pd.notna(maximum) and other_pct > maximum:
            penalty += (other_pct - maximum) / max(maximum, 1.0)

    base_limit = normalize_text(base_detail.get("limit_type", "")).upper()
    candidate_limit = normalize_text(candidate_detail.get("limit_type", "")).upper()
    if base_limit and candidate_limit and base_limit != candidate_limit:
        penalty += 0.10

    shared_bounds = [
        (base_detail.get("min"), candidate_detail.get("min")),
        (base_detail.get("max"), candidate_detail.get("max")),
    ]
    for left, right in shared_bounds:
        if pd.notna(left) and pd.notna(right):
            penalty += 0.10 * abs(left - right) / max(abs(left), 1.0)

    return penalty


def compare_product_against_candidates(
    base_product_id: str,
    candidate_ids: List[str],
    source_repo: Repository,
    target_repo: Repository,
    selected_metrics: List[str],
    metric_weights: Dict[str, float],
    component_weights: Dict[str, float],
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    if not candidate_ids:
        return pd.DataFrame(), {}

    normalized_component_weights = component_weights.copy()
    weight_sum = sum(normalized_component_weights.values())
    if weight_sum <= 0:
        normalized_component_weights = DEFAULT_COMPONENT_WEIGHTS.copy()
        weight_sum = sum(normalized_component_weights.values())
    normalized_component_weights = {
        key: value / weight_sum for key, value in normalized_component_weights.items()
    }

    scale_map = build_scale_map(base_product_id, candidate_ids, source_repo, target_repo, selected_metrics)
    base_metrics = source_repo.nutrient_map.get(base_product_id, {})
    base_ingredients = source_repo.ingredient_detail_map.get(base_product_id, {})
    base_cost = float(source_repo.products.set_index("product_id").loc[base_product_id, "cost_tonne"])

    target_products_indexed = target_repo.products.set_index("product_id")
    ranking_rows: List[dict] = []
    detail_map: Dict[str, dict] = {}

    for candidate_id in candidate_ids:
        candidate_row = target_products_indexed.loc[candidate_id]
        candidate_metrics = target_repo.nutrient_map.get(candidate_id, {})
        candidate_ingredients = target_repo.ingredient_detail_map.get(candidate_id, {})
        candidate_cost = float(candidate_row["cost_tonne"])

        nutrient_impacts = []
        populated_metrics = 0
        for metric in selected_metrics:
            base_value = base_metrics.get(metric, np.nan)
            candidate_value = candidate_metrics.get(metric, np.nan)
            metric_weight = metric_weights.get(metric, 1.0)

            if pd.notna(base_value) and pd.notna(candidate_value):
                normalized_gap = abs(base_value - candidate_value) / max(scale_map.get(metric, 1.0), 1e-6)
                populated_metrics += 1
            else:
                normalized_gap = 2.0

            nutrient_impacts.append(
                {
                    "metric": metric,
                    "actual": base_value,
                    "candidate": candidate_value,
                    "abs_gap": abs(base_value - candidate_value)
                    if pd.notna(base_value) and pd.notna(candidate_value)
                    else np.nan,
                    "normalized_gap": normalized_gap,
                    "weight": metric_weight,
                    "impact": normalized_gap * metric_weight,
                }
            )

        nutrient_score = (
            float(np.average([item["normalized_gap"] for item in nutrient_impacts], weights=[item["weight"] for item in nutrient_impacts]))
            if nutrient_impacts
            else 0.0
        )

        union_ingredients = ingredient_union(base_ingredients, candidate_ingredients)
        ingredient_numerator = 0.0
        ingredient_denominator = 0.0
        ingredient_rows = []
        limit_penalties = []
        shared_ingredients = 0
        for ingredient_name in union_ingredients:
            base_detail = base_ingredients.get(
                ingredient_name,
                {"pct": 0.0, "min": np.nan, "max": np.nan, "limit_type": ""},
            )
            candidate_detail = candidate_ingredients.get(
                ingredient_name,
                {"pct": 0.0, "min": np.nan, "max": np.nan, "limit_type": ""},
            )
            base_pct = base_detail.get("pct", 0.0) or 0.0
            candidate_pct = candidate_detail.get("pct", 0.0) or 0.0
            importance = max(base_pct, candidate_pct, 1.0)
            ingredient_gap = abs(base_pct - candidate_pct)
            ingredient_numerator += ingredient_gap * importance
            ingredient_denominator += importance
            if base_pct > 0 and candidate_pct > 0:
                shared_ingredients += 1

            penalty = compute_limit_penalty(base_detail, candidate_detail)
            limit_penalties.append(penalty)
            ingredient_rows.append(
                {
                    "ingredient": ingredient_name,
                    "actual_pct": base_pct,
                    "candidate_pct": candidate_pct,
                    "abs_gap_pct": ingredient_gap,
                    "actual_min": base_detail.get("min"),
                    "actual_max": base_detail.get("max"),
                    "candidate_min": candidate_detail.get("min"),
                    "candidate_max": candidate_detail.get("max"),
                    "actual_limit": base_detail.get("limit_type", ""),
                    "candidate_limit": candidate_detail.get("limit_type", ""),
                    "limit_penalty": penalty,
                }
            )

        ingredient_score = (ingredient_numerator / ingredient_denominator / 10.0) if ingredient_denominator else 0.0
        limit_score = float(np.nanmean(limit_penalties)) if limit_penalties else 0.0
        price_score = abs(base_cost - candidate_cost) / max(base_cost, 1.0)
        price_diff_pct = 100.0 * (candidate_cost - base_cost) / max(base_cost, 1.0)

        total_score = (
            normalized_component_weights["nutrients"] * nutrient_score
            + normalized_component_weights["ingredients"] * ingredient_score
            + normalized_component_weights["limits"] * limit_score
            + normalized_component_weights["price"] * price_score
        )

        ranking_rows.append(
            {
                "candidate_id": candidate_id,
                "spec_id": candidate_row["spec_id"],
                "product_name": candidate_row["product_name"],
                "display_name": candidate_row["display_name"],
                "species": candidate_row["species"],
                "cost_tonne": candidate_cost,
                "total_score": total_score,
                "nutrient_score": nutrient_score,
                "ingredient_score": ingredient_score,
                "limit_score": limit_score,
                "price_score": price_score,
                "price_diff_pct": price_diff_pct,
                "shared_ingredients": shared_ingredients,
                "metric_coverage": populated_metrics,
            }
        )

        detail_map[candidate_id] = {
            "metric_details": pd.DataFrame(nutrient_impacts).sort_values("impact", ascending=False),
            "ingredient_details": pd.DataFrame(ingredient_rows).sort_values("abs_gap_pct", ascending=False),
        }

    ranking_df = pd.DataFrame(ranking_rows).sort_values("total_score", ascending=True).reset_index(drop=True)
    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    return ranking_df, detail_map


def make_summary_dataframe(repository: Repository) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Archivo": [repository.filename],
            "Productos": [len(repository.products)],
            "Especies detectadas": [repository.products["species"].nunique()],
            "Métricas analíticas": [repository.analyses["analysis_name"].nunique()],
            "Ingredientes únicos": [repository.ingredients["ingredient_label"].nunique()],
        }
    )


def render_repository_debug(repository: Repository, title: str) -> None:
    with st.expander(title, expanded=False):
        st.write("**Productos detectados**")
        st.dataframe(repository.products[["species", "spec_id", "product_name", "cost_tonne"]], use_container_width=True)
        st.write("**Métricas detectadas**")
        metrics = (
            repository.analyses.groupby("analysis_name", dropna=False)
            .size()
            .reset_index(name="frecuencia")
            .sort_values(["frecuencia", "analysis_name"], ascending=[False, True])
        )
        st.dataframe(metrics, use_container_width=True, height=280)


def format_ranking_for_display(ranking_df: pd.DataFrame) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df
    display_df = ranking_df.copy()
    numeric_columns = [
        "total_score",
        "nutrient_score",
        "ingredient_score",
        "limit_score",
        "price_score",
        "price_diff_pct",
        "cost_tonne",
    ]
    for col in numeric_columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(3)
    rename_map = {
        "display_name": "Pienso destino",
        "cost_tonne": "Precio €/t",
        "total_score": "Score total",
        "nutrient_score": "Score nutrientes",
        "ingredient_score": "Score ingredientes",
        "limit_score": "Score límites",
        "price_score": "Score precio",
        "price_diff_pct": "% dif. precio",
        "shared_ingredients": "Ingredientes compartidos",
        "metric_coverage": "Métricas cubiertas",
    }
    existing_rename_map = {key: value for key, value in rename_map.items() if key in display_df.columns}
    return display_df.rename(columns=existing_rename_map)


def render_candidate_detail(
    candidate_id: str,
    ranking_df: pd.DataFrame,
    detail_map: Dict[str, dict],
) -> None:
    candidate_row = ranking_df.set_index("candidate_id").loc[candidate_id]
    metrics_df = detail_map[candidate_id]["metric_details"].copy()
    ingredients_df = detail_map[candidate_id]["ingredient_details"].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score total", f"{candidate_row['total_score']:.3f}")
    col2.metric("Precio destino €/t", f"{candidate_row['cost_tonne']:.3f}", f"{candidate_row['price_diff_pct']:.2f}%")
    col3.metric("Ingredientes compartidos", int(candidate_row["shared_ingredients"]))
    col4.metric("Métricas cubiertas", int(candidate_row["metric_coverage"]))

    detail_tabs = st.tabs(["Métricas", "Ingredientes"])
    with detail_tabs[0]:
        metrics_view = metrics_df.copy()
        for col in ["actual", "candidate", "abs_gap", "normalized_gap", "weight", "impact"]:
            if col in metrics_view.columns:
                metrics_view[col] = metrics_view[col].round(4)
        metrics_view = metrics_view.rename(
            columns={
                "metric": "Métrica",
                "actual": "Actual",
                "candidate": "Destino",
                "abs_gap": "Dif. abs.",
                "normalized_gap": "Dif. normalizada",
                "weight": "Peso",
                "impact": "Impacto",
            }
        )
        st.dataframe(metrics_view, use_container_width=True, height=380)

    with detail_tabs[1]:
        ingredients_view = ingredients_df.copy()
        numeric_cols = [
            "actual_pct",
            "candidate_pct",
            "abs_gap_pct",
            "actual_min",
            "actual_max",
            "candidate_min",
            "candidate_max",
            "limit_penalty",
        ]
        for col in numeric_cols:
            if col in ingredients_view.columns:
                ingredients_view[col] = ingredients_view[col].round(4)
        ingredients_view = ingredients_view.rename(
            columns={
                "ingredient": "Ingrediente",
                "actual_pct": "Actual %",
                "candidate_pct": "Destino %",
                "abs_gap_pct": "Dif. abs. %",
                "actual_min": "Min actual",
                "actual_max": "Max actual",
                "candidate_min": "Min destino",
                "candidate_max": "Max destino",
                "actual_limit": "Lim. actual",
                "candidate_limit": "Lim. destino",
                "limit_penalty": "Penalización límites",
            }
        )
        st.dataframe(ingredients_view, use_container_width=True, height=380)


def build_comparison_report_text(
    source_product_row: pd.Series,
    selected_metrics: List[str],
    metric_weights: Dict[str, float],
    component_weights: Dict[str, float],
    ranking_top: pd.DataFrame,
    detail_map: Dict[str, dict],
    source_repo: Repository,
    target_repo: Repository,
) -> str:
    lines: List[str] = []
    lines.append("INFORME DE COMPARATIVA DE PIENSOS - REBRANDING")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. CONTEXTO DE LA COMPARATIVA")
    lines.append(f"- Archivo origen: {source_repo.filename}")
    lines.append(f"- Archivo destino: {target_repo.filename}")
    lines.append(f"- Especie origen: {source_product_row['species']}")
    lines.append(f"- Producto origen: {source_product_row['product_name']}")
    lines.append(f"- Spec origen: {source_product_row['spec_id']}")
    lines.append(f"- Precio origen €/t: {display_float(float(source_product_row['cost_tonne']), 3)}")
    lines.append("")
    lines.append("2. METODOLOGÍA HEURÍSTICA")
    lines.append("- El ranking ordena candidatos destino por similitud relativa.")
    lines.append("- Un score más bajo indica mayor parecido técnico-económico dentro de esta ejecución.")
    lines.append("- El resultado no sustituye la validación técnica final del equipo.")
    lines.append("")
    lines.append("Pesos globales del score:")
    for component, value in component_weights.items():
        lines.append(f"  * {component}: {value:.2f}")
    lines.append("")
    lines.append("Métricas seleccionadas y peso específico:")
    for metric in selected_metrics:
        lines.append(f"  * {metric}: {metric_weights.get(metric, 1.0):.2f}")
    lines.append("")
    lines.append("3. RANKING DE CANDIDATOS")
    for _, row in ranking_top.iterrows():
        lines.append(
            f"- #{int(row['rank'])} | {row['display_name']} | Score total {row['total_score']:.3f} | "
            f"Precio €/t {row['cost_tonne']:.3f} | Dif. precio {row['price_diff_pct']:.2f}% | "
            f"Ingredientes compartidos {int(row['shared_ingredients'])} | Métricas cubiertas {int(row['metric_coverage'])}"
        )

    for _, row in ranking_top.iterrows():
        candidate_id = row["candidate_id"]
        metric_details = detail_map[candidate_id]["metric_details"].copy()
        ingredient_details = detail_map[candidate_id]["ingredient_details"].copy()

        lines.append("")
        lines.append("-" * 72)
        lines.append(f"4.{int(row['rank'])} DETALLE DEL CANDIDATO #{int(row['rank'])}")
        lines.append(f"Producto destino: {row['display_name']}")
        lines.append(f"Spec destino: {row['spec_id']}")
        lines.append(f"Especie destino: {row['species']}")
        lines.append(f"Precio destino €/t: {row['cost_tonne']:.3f}")
        lines.append(f"Score total: {row['total_score']:.3f}")
        lines.append(f"Score nutrientes: {row['nutrient_score']:.3f}")
        lines.append(f"Score ingredientes: {row['ingredient_score']:.3f}")
        lines.append(f"Score límites: {row['limit_score']:.3f}")
        lines.append(f"Score precio: {row['price_score']:.3f}")
        lines.append(f"Dif. precio %: {row['price_diff_pct']:.2f}%")

        lines.append("")
        lines.append("Principales diferencias analíticas:")
        metric_subset = metric_details.sort_values("impact", ascending=False).head(8)
        if metric_subset.empty:
            lines.append("  * Sin detalle analítico disponible.")
        else:
            for _, metric_row in metric_subset.iterrows():
                lines.append(
                    "  * "
                    f"{metric_row['metric']}: actual {display_float(metric_row['actual'], 4)} | "
                    f"destino {display_float(metric_row['candidate'], 4)} | "
                    f"dif. abs. {display_float(metric_row['abs_gap'], 4)} | "
                    f"impacto {display_float(metric_row['impact'], 4)}"
                )

        lines.append("")
        lines.append("Principales diferencias de ingredientes:")
        ingredient_subset = ingredient_details[ingredient_details['abs_gap_pct'] > 0].sort_values('abs_gap_pct', ascending=False).head(8)
        if ingredient_subset.empty:
            lines.append("  * Sin diferencias relevantes de ingredientes.")
        else:
            for _, ing_row in ingredient_subset.iterrows():
                lines.append(
                    "  * "
                    f"{ing_row['ingredient']}: actual {display_float(ing_row['actual_pct'], 3)}% | "
                    f"destino {display_float(ing_row['candidate_pct'], 3)}% | "
                    f"dif. abs. {display_float(ing_row['abs_gap_pct'], 3)}% | "
                    f"penalización límites {display_float(ing_row['limit_penalty'], 3)}"
                )

        limit_alerts = ingredient_details[ingredient_details['limit_penalty'] > 0].sort_values('limit_penalty', ascending=False).head(6)
        lines.append("")
        lines.append("Alertas de límites:")
        if limit_alerts.empty:
            lines.append("  * No se han detectado alertas de límites en los ingredientes revisados.")
        else:
            for _, alert_row in limit_alerts.iterrows():
                lines.append(
                    "  * "
                    f"{alert_row['ingredient']}: límite actual {alert_row['actual_limit']} "
                    f"[{display_float(alert_row['actual_min'], 3)} - {display_float(alert_row['actual_max'], 3)}] | "
                    f"límite destino {alert_row['candidate_limit']} "
                    f"[{display_float(alert_row['candidate_min'], 3)} - {display_float(alert_row['candidate_max'], 3)}] | "
                    f"penalización {display_float(alert_row['limit_penalty'], 3)}"
                )

    lines.append("")
    lines.append("5. CONCLUSIÓN OPERATIVA")
    lines.append("- Este informe sirve para priorizar equivalencias técnicas y reducir revisión manual.")
    lines.append("- La decisión final debe confirmar ajuste nutricional, viabilidad industrial, coste y encaje comercial.")
    lines.append("- Se recomienda validar en detalle los primeros candidatos del ranking antes de migrar el SKU.")
    return "\n".join(lines)



def sanitize_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\\/*?:\[\]]", "_", str(name))
    cleaned = cleaned[:31].strip()
    return cleaned or "Hoja"


def autosize_openpyxl_worksheet(worksheet) -> None:
    for column_cells in worksheet.columns:
        values = ["" if cell.value is None else str(cell.value) for cell in column_cells]
        max_length = max((len(value) for value in values), default=0)
        column_letter = column_cells[0].column_letter
        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 40)


def build_excel_export_bytes(
    source_product_row: pd.Series,
    ranking_df: pd.DataFrame,
    ranking_top: pd.DataFrame,
    detail_map: Dict[str, dict],
    selected_metrics: List[str],
    metric_weights: Dict[str, float],
    component_weights: Dict[str, float],
    source_repo: Repository,
    target_repo: Repository,
    include_full_ranking: bool = True,
) -> bytes:
    output = io.BytesIO()

    resumen_df = pd.DataFrame(
        [
            {"Campo": "Archivo origen", "Valor": source_repo.filename},
            {"Campo": "Archivo destino", "Valor": target_repo.filename},
            {"Campo": "Especie origen", "Valor": source_product_row["species"]},
            {"Campo": "Producto origen", "Valor": source_product_row["product_name"]},
            {"Campo": "Spec origen", "Valor": source_product_row["spec_id"]},
            {"Campo": "Precio origen €/t", "Valor": float(source_product_row["cost_tonne"])},
        ]
    )

    component_weights_df = pd.DataFrame(
        [{"Componente": key, "Peso": value} for key, value in component_weights.items()]
    )
    metric_weights_df = pd.DataFrame(
        [{"Métrica": metric, "Peso": metric_weights.get(metric, 1.0)} for metric in selected_metrics]
    )

    ranking_export_df = ranking_df.copy() if include_full_ranking else ranking_top.copy()
    if "candidate_id" in ranking_export_df.columns:
        ranking_export_df = ranking_export_df.drop(columns=["candidate_id"])
    ranking_export_df = ranking_export_df.rename(
        columns={
            "display_name": "producto_destino",
            "cost_tonne": "precio_eur_t",
            "total_score": "score_total",
            "nutrient_score": "score_nutrientes",
            "ingredient_score": "score_ingredientes",
            "limit_score": "score_limites",
            "price_score": "score_precio",
            "price_diff_pct": "dif_precio_pct",
            "shared_ingredients": "ingredientes_compartidos",
            "metric_coverage": "metricas_cubiertas",
        }
    )

    report_text = build_comparison_report_text(
        source_product_row=source_product_row,
        selected_metrics=selected_metrics,
        metric_weights=metric_weights,
        component_weights=component_weights,
        ranking_top=ranking_top,
        detail_map=detail_map,
        source_repo=source_repo,
        target_repo=target_repo,
    )
    report_df = pd.DataFrame({"Informe": report_text.splitlines()})

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        resumen_df.to_excel(writer, sheet_name="Resumen", index=False, startrow=0)
        component_weights_df.to_excel(writer, sheet_name="Resumen", index=False, startrow=len(resumen_df) + 3)
        metric_weights_df.to_excel(writer, sheet_name="Resumen", index=False, startrow=len(resumen_df) + len(component_weights_df) + 7)

        ranking_export_df.to_excel(writer, sheet_name="Ranking", index=False)
        report_df.to_excel(writer, sheet_name="Informe", index=False)

        for _, row in ranking_top.iterrows():
            candidate_id = row["candidate_id"]
            rank_label = f"{int(row['rank']):02d}"
            metrics_df = detail_map[candidate_id]["metric_details"].copy()
            ingredients_df = detail_map[candidate_id]["ingredient_details"].copy()

            metrics_sheet = sanitize_sheet_name(f"{rank_label}_metricas")
            ingredients_sheet = sanitize_sheet_name(f"{rank_label}_ingredientes")

            metrics_df.to_excel(writer, sheet_name=metrics_sheet, index=False)
            ingredients_df.to_excel(writer, sheet_name=ingredients_sheet, index=False)

            workbook = writer.book
            worksheet = workbook[metrics_sheet]
            worksheet["J1"] = "Producto destino"
            worksheet["J2"] = row["display_name"]
            worksheet["J3"] = "Score total"
            worksheet["J4"] = float(row["total_score"])
            worksheet["J5"] = "Precio €/t"
            worksheet["J6"] = float(row["cost_tonne"])

            worksheet2 = workbook[ingredients_sheet]
            worksheet2["L1"] = "Producto destino"
            worksheet2["L2"] = row["display_name"]
            worksheet2["L3"] = "Spec destino"
            worksheet2["L4"] = row["spec_id"]
            worksheet2["L5"] = "Dif. precio %"
            worksheet2["L6"] = float(row["price_diff_pct"])

        workbook = writer.book
        for worksheet in workbook.worksheets:
            autosize_openpyxl_worksheet(worksheet)
            worksheet.freeze_panes = "A2"

    output.seek(0)
    return output.getvalue()


def repository_selector(label: str, repository: Repository, default_species: str | None = None) -> Tuple[str, pd.DataFrame]:
    species_options = sorted(repository.products["species"].dropna().unique().tolist())
    if not species_options:
        species_options = ["Sin clasificar"]

    species_options = ["Todas"] + species_options
    default_index = species_options.index(default_species) if default_species in species_options else 0
    selected_species = st.selectbox(label, species_options, index=default_index)
    if selected_species == "Todas":
        filtered_products = repository.products.copy()
    else:
        filtered_products = repository.products[repository.products["species"] == selected_species].copy()
    return selected_species, filtered_products


st.title("Comparador técnico de fórmulas | Rebranding")
st.caption(
    "La aplicación lee exportaciones tipo Multi-Mix y Single-Mix del Excel, reconstruye cada fórmula y propone los productos destino más parecidos."
)
st.info(
    "El ranking de similitud es una ayuda de priorización. Ordena candidatos por cercanía nutricional, ingredientes, límites y precio, pero la decisión final sigue siendo técnica."
)

with st.sidebar:
    st.header("1) Cargar archivos")
    source_file = st.file_uploader("Archivo origen (portfolio actual)", type=["xlsx"])
    target_file = st.file_uploader("Archivo destino (gama estándar)", type=["xlsx"])

    st.header("2) Parámetros del ranking")
    top_n = st.slider("Número de candidatos", min_value=3, max_value=10, value=5, step=1)
    component_weights = {
        "nutrients": st.slider("Peso nutrientes", 0.0, 1.0, float(DEFAULT_COMPONENT_WEIGHTS["nutrients"]), 0.05),
        "ingredients": st.slider("Peso ingredientes", 0.0, 1.0, float(DEFAULT_COMPONENT_WEIGHTS["ingredients"]), 0.05),
        "limits": st.slider("Peso límites", 0.0, 1.0, float(DEFAULT_COMPONENT_WEIGHTS["limits"]), 0.05),
        "price": st.slider("Peso precio", 0.0, 1.0, float(DEFAULT_COMPONENT_WEIGHTS["price"]), 0.05),
    }

    st.header("3) Ayuda")
    if st.button("Mostrar / ocultar README"):
        st.session_state["show_readme"] = not st.session_state.get("show_readme", False)
    st.download_button(
        label="Descargar README.md",
        data=load_readme_text().encode("utf-8"),
        file_name="README.md",
        mime="text/markdown",
    )
    with st.expander("Cómo leer el análisis heurístico", expanded=False):
        st.markdown(heuristic_help_markdown())

if st.session_state.get("show_readme", False):
    with st.expander("README / guía rápida de uso", expanded=True):
        st.markdown(load_readme_text())

if not source_file or not target_file:
    st.warning("Sube los dos archivos Excel para empezar.")
    st.stop()

try:
    source_repo = parse_repository(source_file.getvalue(), source_file.name)
    target_repo = parse_repository(target_file.getvalue(), target_file.name)
except Exception as exc:
    st.error(f"No se han podido interpretar los archivos: {exc}")
    st.stop()

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.subheader("Archivo origen")
    st.dataframe(make_summary_dataframe(source_repo), use_container_width=True, hide_index=True)
with summary_col2:
    st.subheader("Archivo destino")
    st.dataframe(make_summary_dataframe(target_repo), use_container_width=True, hide_index=True)

st.divider()

sel_col1, sel_col2 = st.columns([1, 1])
with sel_col1:
    selected_source_species, source_products_filtered = repository_selector("Especie origen", source_repo)
    source_product_id = st.selectbox(
        "Pienso actual",
        source_products_filtered["product_id"].tolist(),
        format_func=lambda pid: source_products_filtered.set_index("product_id").loc[pid, "display_name"],
    )

with sel_col2:
    selected_target_species, target_products_filtered = repository_selector(
        "Especie destino", target_repo, default_species=selected_source_species
    )
    compare_all_target_species = st.checkbox("Comparar contra toda la gama destino", value=False)
    if compare_all_target_species:
        target_products_filtered = target_repo.products.copy()

source_product_row = source_repo.products.set_index("product_id").loc[source_product_id]
source_metrics = source_repo.analyses[source_repo.analyses["product_id"] == source_product_id][
    ["analysis_name", "level", "min", "max", "limit_type"]
].copy()

candidate_ids = [pid for pid in target_products_filtered["product_id"].tolist() if pid != source_product_id]
common_metrics = sorted(
    set(source_repo.nutrient_map.get(source_product_id, {}).keys())
    & set(target_repo.analyses["analysis_name"].dropna().unique().tolist())
)

if not common_metrics:
    st.error("No hay métricas analíticas comunes entre el producto origen y la gama destino filtrada.")
    st.stop()

saved_metrics, saved_metric_weights = get_metric_preferences(selected_source_species, common_metrics)
species_key = species_pref_key(selected_source_species)
metric_multiselect_key = f"selected_metrics_{species_key}"

if metric_multiselect_key not in st.session_state:
    st.session_state[metric_multiselect_key] = saved_metrics
else:
    filtered_metrics = [metric for metric in st.session_state[metric_multiselect_key] if metric in common_metrics]
    st.session_state[metric_multiselect_key] = filtered_metrics or saved_metrics

st.subheader("Selección de métricas críticas")
metric_selector_col1, metric_selector_col2 = st.columns([1.4, 1])
with metric_selector_col1:
    st.caption(
        "La selección de métricas y sus pesos se conserva por especie durante la sesión para no tener que redefinirla en cada comparación."
    )
    if st.button("Restaurar selección automática recomendada"):
        reset_metric_preferences(selected_source_species)
        st.rerun()

    selected_metrics = st.multiselect(
        "Métricas incluidas en la comparación",
        options=common_metrics,
        key=metric_multiselect_key,
        help="Puedes ajustar manualmente qué nutrientes/analíticas entran en el score. La app conservará esta selección al cambiar de pienso dentro de la misma especie.",
    )
with metric_selector_col2:
    st.write("**Producto origen**")
    source_info_df = pd.DataFrame(
        {
            "Campo": ["Especie", "Producto", "Spec", "Precio €/t"],
            "Valor": [
                source_product_row["species"],
                source_product_row["product_name"],
                source_product_row["spec_id"],
                display_float(float(source_product_row["cost_tonne"]), 3),
            ],
        }
    )
    st.dataframe(source_info_df, use_container_width=True, hide_index=True)

if not selected_metrics:
    st.warning("Selecciona al menos una métrica para lanzar la comparación.")
    st.stop()

weight_editor_df = pd.DataFrame(
    {
        "Métrica": selected_metrics,
        "Peso": [saved_metric_weights.get(metric, default_metric_weight(metric, selected_source_species)) for metric in selected_metrics],
    }
)

st.write("**Pesos por métrica**")
weights_editor_key = f"weights_editor_{species_key}_{stable_metrics_signature(selected_metrics)}"
edited_weights_df = st.data_editor(
    weight_editor_df,
    key=weights_editor_key,
    use_container_width=True,
    hide_index=True,
    disabled=["Métrica"],
    column_config={"Peso": st.column_config.NumberColumn(min_value=0.1, max_value=10.0, step=0.1)},
)
metric_weights = dict(zip(edited_weights_df["Métrica"], edited_weights_df["Peso"]))
save_metric_preferences(selected_source_species, selected_metrics, metric_weights)

ranking_df, detail_map = compare_product_against_candidates(
    base_product_id=source_product_id,
    candidate_ids=candidate_ids,
    source_repo=source_repo,
    target_repo=target_repo,
    selected_metrics=selected_metrics,
    metric_weights=metric_weights,
    component_weights=component_weights,
)

if ranking_df.empty:
    st.warning("No se han encontrado candidatos destino con la configuración actual.")
    st.stop()

ranking_top = ranking_df.head(top_n).copy()
display_ranking = format_ranking_for_display(ranking_top)

result_tabs = st.tabs(["Ranking", "Detalle comparativo", "Datos parseados"])

with result_tabs[0]:
    st.subheader(f"Top {top_n} equivalencias propuestas")
    st.dataframe(display_ranking, use_container_width=True, hide_index=True)

    export_df = ranking_top[[
        "rank",
        "display_name",
        "spec_id",
        "species",
        "cost_tonne",
        "total_score",
        "nutrient_score",
        "ingredient_score",
        "limit_score",
        "price_diff_pct",
        "shared_ingredients",
        "metric_coverage",
    ]].copy()
    export_df.columns = [
        "rank",
        "producto_destino",
        "spec_id",
        "species",
        "precio_eur_t",
        "score_total",
        "score_nutrientes",
        "score_ingredientes",
        "score_limites",
        "dif_precio_pct",
        "ingredientes_compartidos",
        "metricas_cubiertas",
    ]
    excel_bytes = build_excel_export_bytes(
        source_product_row=source_product_row,
        ranking_df=ranking_df,
        ranking_top=ranking_top,
        detail_map=detail_map,
        selected_metrics=selected_metrics,
        metric_weights=metric_weights,
        component_weights=component_weights,
        source_repo=source_repo,
        target_repo=target_repo,
        include_full_ranking=True,
    )
    st.download_button(
        label="Descargar resultados en Excel",
        data=excel_bytes,
        file_name=f"comparativa_rebranding_{source_product_row['spec_id']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    report_text = build_comparison_report_text(
        source_product_row=source_product_row,
        selected_metrics=selected_metrics,
        metric_weights=metric_weights,
        component_weights=component_weights,
        ranking_top=ranking_top,
        detail_map=detail_map,
        source_repo=source_repo,
        target_repo=target_repo,
    )
    st.download_button(
        label="Descargar informe comparativo en TXT",
        data=report_text.encode("utf-8-sig"),
        file_name=f"informe_comparativo_{source_product_row['spec_id']}.txt",
        mime="text/plain",
    )

with result_tabs[1]:
    candidate_choice = st.selectbox(
        "Selecciona un candidato para ver el detalle",
        ranking_top["candidate_id"].tolist(),
        format_func=lambda pid: ranking_top.set_index("candidate_id").loc[pid, "display_name"],
    )
    render_candidate_detail(candidate_choice, ranking_df, detail_map)

with result_tabs[2]:
    render_repository_debug(source_repo, "Depuración archivo origen")
    render_repository_debug(target_repo, "Depuración archivo destino")
    with st.expander("Métricas del producto origen", expanded=False):
        st.dataframe(source_metrics, use_container_width=True, hide_index=True, height=380)
