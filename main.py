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
from openpyxl.utils import get_column_letter

st.set_page_config(
    page_title="Comparador de fórmulas | Rebranding",
    page_icon="📊",
    layout="wide",
)

NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$|^\.$")
SPEC_LINE_RE = re.compile(
    r"Specification:\s*([^\s]+)\s+(.+?)\s*:\s*Cost/tonne:\s*([0-9.]+)",
    re.IGNORECASE,
)
SP_LINE_RE = re.compile(
    r"\bSP:\s*([^\s]+)\s+(.+?)\s+\d+(?:[.,]\d+)?\s*%\s*,\s*\d+(?:[.,]\d+)?\s*Kg\s*"
    r"\((?:Re)?cost:\s*([0-9.]+)\).*?(?:Optimal cost:\s*([0-9.]+))?",
    re.IGNORECASE,
)

LIMIT_TOKENS = {"MIN", "MAX", "FIX", "EQL", "EQ", "RNG", "MINMAX"}

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
            " ib ",
            " iber",
            "optiporc",
            "baby",
            "cochin",
            "prestarter",
            "starter porc",
            "reproductoras",
            "gestantes",
            "lactantes",
            "gestacion",
            "recria fr",
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
    return re.sub(r"\s+", " ", str(text or "")).strip(" :")


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
    if value is None:
        return ""
    try:
        if math.isnan(float(value)):
            return ""
    except Exception:
        pass
    return f"{float(value):.{decimals}f}"


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

La aplicación construye un **ranking heurístico** para ayudarte a priorizar qué referencias estándar merecen revisión técnica.

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
    text = f" {filename} {product_name} ".lower()
    for label, patterns in SPECIES_RULES:
        if any(pattern in text for pattern in patterns):
            return label
    return "Sin clasificar"


@st.cache_data(show_spinner=False)
def read_lines_from_excel(file_bytes: bytes, filename: str) -> Tuple[List[str], str]:
    workbook = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None, dtype=str)
    if not workbook:
        raise ValueError(f"No se ha podido leer ninguna hoja en {filename}.")

    first_sheet_name = list(workbook.keys())[0]
    sheet = workbook[first_sheet_name].fillna("")
    lines: List[str] = []
    for _, row in sheet.iterrows():
        parts = [normalize_text(cell) for cell in row.tolist() if normalize_text(cell)]
        lines.append(" ".join(parts) if parts else "")
    return lines, first_sheet_name


def split_formula_blocks(lines: List[str]) -> List[List[str]]:
    starts = [idx for idx, line in enumerate(lines) if parse_spec_line(line)]
    if not starts:
        raise ValueError(
            "No se han encontrado bloques tipo 'Specification:' ni 'SP:'. El Excel no parece tener el mismo formato que el ejemplo."
        )

    blocks: List[List[str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines) - 1
        blocks.append(lines[start : end + 1])
    return blocks


def parse_spec_line(line: str) -> dict:
    text = normalize_text(line)
    multi = SPEC_LINE_RE.search(text)
    if multi:
        raw_product = normalize_text(multi.group(2))
        code, product_name = split_product_code_and_name(raw_product)
        return {
            "spec_id": multi.group(1),
            "product_raw": raw_product,
            "product_code": code,
            "product_name": product_name,
            "cost_tonne": safe_float(multi.group(3)),
            "format_type": "multi_mix",
        }

    single = SP_LINE_RE.search(text)
    if single:
        raw_product = normalize_text(single.group(2))
        code, product_name = split_product_code_and_name(raw_product)
        cost_value = single.group(4) if single.group(4) not in {None, ""} else single.group(3)
        return {
            "spec_id": single.group(1),
            "product_raw": raw_product,
            "product_code": code,
            "product_name": product_name,
            "cost_tonne": safe_float(cost_value),
            "format_type": "single_mix",
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
        "sensitivity_start": None,
        "rejected_start": None,
    }
    for idx, line in enumerate(block_lines):
        upper = line.upper()
        if "INCLUDED RAW MATERIALS" in upper:
            positions["ingredients_start"] = idx + 2
        elif upper.strip().startswith("ANALYSIS") or "NUTRIENT ANALYSIS" in upper:
            positions["analysis_start"] = idx + 2
        elif "REJECTED RAW MATERIALS" in upper:
            positions["rejected_start"] = idx
        elif "RAW MATERIAL SENSITIVITY" in upper:
            positions["sensitivity_start"] = idx
            break
    return positions


def parse_ingredient_line(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 5:
        return None

    pct_idx = None
    for i in range(2, len(tokens) - 2):
        if token_is_numeric(tokens[i]) and token_is_numeric(tokens[i + 1]) and token_is_numeric(tokens[i + 2]):
            first = safe_float(tokens[i])
            second = safe_float(tokens[i + 1])
            third = safe_float(tokens[i + 2])
            if pd.notna(first) and pd.notna(second) and pd.notna(third) and second > first and third > 10:
                pct_idx = i
                break
    if pct_idx is None:
        for i in range(2, len(tokens)):
            if token_is_numeric(tokens[i]):
                pct_idx = i
                break
    if pct_idx is None or pct_idx + 2 >= len(tokens):
        return None

    ingredient_key = normalize_text(tokens[0])
    ingredient_name = normalize_text(" ".join(tokens[1:pct_idx]))
    ingredient_label = normalize_text(f"{ingredient_key} {ingredient_name}")

    pct = safe_float(tokens[pct_idx])
    kilos = safe_float(tokens[pct_idx + 1]) if pct_idx + 1 < len(tokens) else np.nan
    avg_cost = safe_float(tokens[pct_idx + 2]) if pct_idx + 2 < len(tokens) else np.nan

    cursor = pct_idx + 3
    limit_token = ""
    if cursor < len(tokens) and tokens[cursor].upper() in LIMIT_TOKENS:
        limit_token = tokens[cursor].upper()
        cursor += 1

    minimum = safe_float(tokens[cursor]) if cursor < len(tokens) else np.nan
    maximum = safe_float(tokens[cursor + 1]) if cursor + 1 < len(tokens) else np.nan

    return {
        "ingredient_key": ingredient_key,
        "ingredient_name": ingredient_name,
        "ingredient_label": ingredient_label,
        "avg_cost": avg_cost,
        "pct": pct,
        "kilos": kilos,
        "tonnes": np.nan,
        "limit_type": limit_token,
        "min": minimum,
        "max": maximum,
    }


def parse_analysis_line(line: str) -> dict | None:
    tokens = line.split()
    if len(tokens) < 2:
        return None

    if tokens[:3] == ["[", "PESO", "]"]:
        analysis_name = "[PESO]"
        start_idx = 3
    else:
        analysis_name = normalize_text(tokens[0])
        start_idx = 1

    level_idx = None
    for i in range(start_idx, len(tokens)):
        if token_is_numeric(tokens[i]):
            level_idx = i
            break
    if level_idx is None:
        return None

    level = safe_float(tokens[level_idx])
    cursor = level_idx + 1

    limit_type = ""
    if cursor < len(tokens) and tokens[cursor].upper() in LIMIT_TOKENS:
        limit_type = tokens[cursor].upper()
        cursor += 1

    minimum = safe_float(tokens[cursor]) if cursor < len(tokens) else np.nan
    maximum = safe_float(tokens[cursor + 1]) if cursor + 1 < len(tokens) else np.nan
    without_dummies = safe_float(tokens[-1]) if tokens else np.nan

    return {
        "analysis_name": analysis_name,
        "level": level,
        "limit_type": limit_type,
        "min": minimum,
        "max": maximum,
        "without_dummies": without_dummies,
    }


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
                "format_type": spec_meta.get("format_type", ""),
            }
        )

        if positions["ingredients_start"] is not None:
            if positions["rejected_start"] is not None:
                ingredient_end = positions["rejected_start"]
            elif positions["analysis_start"] is not None:
                ingredient_end = positions["analysis_start"] - 2
            else:
                ingredient_end = len(block)

            ingredient_lines = block[positions["ingredients_start"] : max(positions["ingredients_start"], ingredient_end)]
            for line in ingredient_lines:
                stripped = line.strip()
                upper = stripped.upper()
                if (
                    not stripped
                    or stripped.startswith("-")
                    or "REJECTED RAW MATERIALS" in upper
                    or "RAW MATERIAL SENSITIVITY" in upper
                ):
                    continue
                parsed = parse_ingredient_line(line)
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
                    or "RAW MATERIAL SENSITIVITY" in upper
                    or "REJECTED RAW MATERIALS" in upper
                ):
                    continue
                parsed = parse_analysis_line(line)
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
    normalized_component_weights = {key: value / weight_sum for key, value in normalized_component_weights.items()}

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
            float(
                np.average(
                    [item["normalized_gap"] for item in nutrient_impacts],
                    weights=[item["weight"] for item in nutrient_impacts],
                )
            )
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
            "Formato detectado": [", ".join(sorted(repository.products["format_type"].dropna().astype(str).unique().tolist()))],
        }
    )


def render_repository_debug(repository: Repository, title: str) -> None:
    with st.expander(title, expanded=False):
        st.write("**Productos detectados**")
        cols = [c for c in ["species", "spec_id", "product_name", "cost_tonne", "format_type"] if c in repository.products.columns]
        st.dataframe(repository.products[cols], use_container_width=True)
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


def render_candidate_detail(candidate_id: str, ranking_df: pd.DataFrame, detail_map: Dict[str, dict]) -> None:
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
        ingredient_subset = ingredient_details[ingredient_details["abs_gap_pct"] > 0].sort_values("abs_gap_pct", ascending=False).head(8)
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

        limit_alerts = ingredient_details[ingredient_details["limit_penalty"] > 0].sort_values("limit_penalty", ascending=False).head(6)
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


def build_excel_export(
    source_product_row: pd.Series,
    ranking_top: pd.DataFrame,
    detail_map: Dict[str, dict],
    report_text: str,
) -> bytes:
    ranking_export = ranking_top[
        [
            "rank",
            "display_name",
            "spec_id",
            "species",
            "cost_tonne",
            "total_score",
            "nutrient_score",
            "ingredient_score",
            "limit_score",
            "price_score",
            "price_diff_pct",
            "shared_ingredients",
            "metric_coverage",
        ]
    ].copy()
    ranking_export.columns = [
        "rank",
        "producto_destino",
        "spec_id",
        "species",
        "precio_eur_t",
        "score_total",
        "score_nutrientes",
        "score_ingredientes",
        "score_limites",
        "score_precio",
        "dif_precio_pct",
        "ingredientes_compartidos",
        "metricas_cubiertas",
    ]

    resumen = pd.DataFrame(
        {
            "Campo": ["Producto origen", "Spec origen", "Especie", "Precio origen €/t", "Candidatos exportados"],
            "Valor": [
                source_product_row["product_name"],
                source_product_row["spec_id"],
                source_product_row["species"],
                float(source_product_row["cost_tonne"]),
                len(ranking_top),
            ],
        }
    )
    informe_df = pd.DataFrame({"Informe": report_text.splitlines()})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        ranking_export.to_excel(writer, sheet_name="Ranking", index=False)
        informe_df.to_excel(writer, sheet_name="Informe", index=False)

        for _, row in ranking_top.iterrows():
            rank = int(row["rank"])
            metric_sheet = f"Metricas_{rank}"[:31]
            ingr_sheet = f"Ingredientes_{rank}"[:31]

            metrics_df = detail_map[row["candidate_id"]]["metric_details"].copy()
            ingredients_df = detail_map[row["candidate_id"]]["ingredient_details"].copy()
            metrics_df.to_excel(writer, sheet_name=metric_sheet, index=False)
            ingredients_df.to_excel(writer, sheet_name=ingr_sheet, index=False)

            ws_m = writer.book[metric_sheet]
            ws_m["J1"] = "Candidato"
            ws_m["J2"] = row["display_name"]
            ws_i = writer.book[ingr_sheet]
            ws_i["L1"] = "Candidato"
            ws_i["L2"] = row["display_name"]

        for ws in writer.book.worksheets:
            for col_cells in ws.columns:
                max_len = 0
                col_letter = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        max_len = max(max_len, len(str(cell.value or "")))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)

    buffer.seek(0)
    return buffer.getvalue()


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




def compare_multiple_source_products(
    source_product_ids: List[str],
    candidate_ids: List[str],
    source_repo: Repository,
    target_repo: Repository,
    selected_metrics: List[str],
    metric_weights: Dict[str, float],
    component_weights: Dict[str, float],
    top_n: int,
) -> Tuple[Dict[str, dict], pd.DataFrame]:
    bulk_results: Dict[str, dict] = {}
    consolidated_rows: List[dict] = []

    source_products_indexed = source_repo.products.set_index("product_id")
    for source_product_id in source_product_ids:
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
            continue

        ranking_top = ranking_df.head(top_n).copy()
        source_row = source_products_indexed.loc[source_product_id]
        report_text = build_comparison_report_text(
            source_product_row=source_row,
            selected_metrics=selected_metrics,
            metric_weights=metric_weights,
            component_weights=component_weights,
            ranking_top=ranking_top,
            detail_map=detail_map,
            source_repo=source_repo,
            target_repo=target_repo,
        )

        bulk_results[source_product_id] = {
            "source_product_row": source_row,
            "ranking_df": ranking_df,
            "ranking_top": ranking_top,
            "detail_map": detail_map,
            "report_text": report_text,
        }

        for _, row in ranking_top.iterrows():
            consolidated_rows.append(
                {
                    "source_product_id": source_product_id,
                    "source_display_name": source_row["display_name"],
                    "source_product_name": source_row["product_name"],
                    "source_spec_id": source_row["spec_id"],
                    "source_species": source_row["species"],
                    "source_price_eur_t": float(source_row["cost_tonne"]),
                    "rank": int(row["rank"]),
                    "candidate_id": row["candidate_id"],
                    "candidate_display_name": row["display_name"],
                    "candidate_product_name": row["product_name"],
                    "candidate_spec_id": row["spec_id"],
                    "candidate_species": row["species"],
                    "candidate_price_eur_t": float(row["cost_tonne"]),
                    "score_total": float(row["total_score"]),
                    "score_nutrientes": float(row["nutrient_score"]),
                    "score_ingredientes": float(row["ingredient_score"]),
                    "score_limites": float(row["limit_score"]),
                    "score_precio": float(row["price_score"]),
                    "dif_precio_pct": float(row["price_diff_pct"]),
                    "ingredientes_compartidos": int(row["shared_ingredients"]),
                    "metricas_cubiertas": int(row["metric_coverage"]),
                }
            )

    consolidated_df = pd.DataFrame(consolidated_rows)
    if not consolidated_df.empty:
        consolidated_df = consolidated_df.sort_values(["source_display_name", "rank"]).reset_index(drop=True)
    return bulk_results, consolidated_df


def build_bulk_selection_tables(
    selected_choice_map: Dict[str, str],
    bulk_results: Dict[str, dict],
    source_repo: Repository,
    target_repo: Repository,
    selected_metrics: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[dict] = []
    nutrient_rows_wide: List[dict] = []
    metric_rows_long: List[dict] = []
    ingredient_rows_long: List[dict] = []

    target_products_indexed = target_repo.products.set_index("product_id")

    for source_product_id, candidate_id in selected_choice_map.items():
        result = bulk_results.get(source_product_id)
        if not result or candidate_id not in result["ranking_df"].set_index("candidate_id").index:
            continue

        source_row = result["source_product_row"]
        ranking_row = result["ranking_df"].set_index("candidate_id").loc[candidate_id]
        candidate_row = target_products_indexed.loc[candidate_id]

        source_price = float(source_row["cost_tonne"])
        candidate_price = float(candidate_row["cost_tonne"])
        price_diff_pct = 100.0 * (candidate_price - source_price) / max(source_price, 1.0)

        summary_rows.append(
            {
                "producto_origen": source_row["display_name"],
                "spec_origen": source_row["spec_id"],
                "especie_origen": source_row["species"],
                "precio_origen_eur_t": source_price,
                "producto_destino_elegido": candidate_row["display_name"],
                "spec_destino": candidate_row["spec_id"],
                "especie_destino": candidate_row["species"],
                "precio_destino_eur_t": candidate_price,
                "dif_precio_pct": price_diff_pct,
                "score_total": float(ranking_row["total_score"]),
                "score_nutrientes": float(ranking_row["nutrient_score"]),
                "score_ingredientes": float(ranking_row["ingredient_score"]),
                "score_limites": float(ranking_row["limit_score"]),
                "score_precio": float(ranking_row["price_score"]),
                "ingredientes_compartidos": int(ranking_row["shared_ingredients"]),
                "metricas_cubiertas": int(ranking_row["metric_coverage"]),
            }
        )

        nutrient_wide_row = {
            "producto_origen": source_row["display_name"],
            "producto_destino_elegido": candidate_row["display_name"],
            "precio_origen_eur_t": source_price,
            "precio_destino_eur_t": candidate_price,
            "dif_precio_pct": price_diff_pct,
            "score_total": float(ranking_row["total_score"]),
        }
        source_metrics = source_repo.nutrient_map.get(source_product_id, {})
        candidate_metrics = target_repo.nutrient_map.get(candidate_id, {})
        for metric in selected_metrics:
            source_value = source_metrics.get(metric, np.nan)
            candidate_value = candidate_metrics.get(metric, np.nan)
            diff_value = candidate_value - source_value if pd.notna(source_value) and pd.notna(candidate_value) else np.nan
            nutrient_wide_row[f"{metric}_origen"] = source_value
            nutrient_wide_row[f"{metric}_destino"] = candidate_value
            nutrient_wide_row[f"{metric}_dif"] = diff_value
            metric_rows_long.append(
                {
                    "producto_origen": source_row["display_name"],
                    "producto_destino_elegido": candidate_row["display_name"],
                    "metrica": metric,
                    "valor_origen": source_value,
                    "valor_destino": candidate_value,
                    "diferencia_destino_menos_origen": diff_value,
                }
            )
        nutrient_rows_wide.append(nutrient_wide_row)

        ingredients_df = result["detail_map"][candidate_id]["ingredient_details"].copy()
        if not ingredients_df.empty:
            ingredients_df.insert(0, "producto_origen", source_row["display_name"])
            ingredients_df.insert(1, "producto_destino_elegido", candidate_row["display_name"])
            ingredient_rows_long.extend(ingredients_df.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    nutrient_wide_df = pd.DataFrame(nutrient_rows_wide)
    metric_long_df = pd.DataFrame(metric_rows_long)
    ingredient_long_df = pd.DataFrame(ingredient_rows_long)
    return summary_df, nutrient_wide_df, metric_long_df, ingredient_long_df




def build_bulk_matrix_editor_df(
    source_product_ids: List[str],
    bulk_results: Dict[str, dict],
    top_n: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    max_candidates = 0
    for source_product_id in source_product_ids:
        result = bulk_results.get(source_product_id)
        if not result:
            continue
        ranking_top = result["ranking_top"].copy().reset_index(drop=True)
        source_row = result["source_product_row"]
        max_candidates = max(max_candidates, len(ranking_top))
        row = {
            "source_product_id": source_product_id,
            "producto_origen": source_row["display_name"],
            "precio_origen_eur_t": float(source_row["cost_tonne"]),
            "opcion_elegida": 1,
        }
        for idx, candidate in ranking_top.iterrows():
            rank_no = idx + 1
            row[f"opcion_{rank_no}"] = candidate["display_name"]
            row[f"score_{rank_no}"] = float(candidate["total_score"])
            row[f"precio_{rank_no}_eur_t"] = float(candidate["cost_tonne"])
            row[f"dif_precio_{rank_no}_pct"] = float(candidate["price_diff_pct"])
        rows.append(row)

    matrix_df = pd.DataFrame(rows)
    if matrix_df.empty:
        return matrix_df

    for rank_no in range(1, max_candidates + 1):
        for col in [f"opcion_{rank_no}", f"score_{rank_no}", f"precio_{rank_no}_eur_t", f"dif_precio_{rank_no}_pct"]:
            if col not in matrix_df.columns:
                matrix_df[col] = np.nan if col != f"opcion_{rank_no}" else ""

    ordered_cols = ["source_product_id", "producto_origen", "precio_origen_eur_t", "opcion_elegida"]
    for rank_no in range(1, max_candidates + 1):
        ordered_cols.extend([
            f"opcion_{rank_no}",
            f"score_{rank_no}",
            f"precio_{rank_no}_eur_t",
            f"dif_precio_{rank_no}_pct",
        ])
    return matrix_df[ordered_cols]


def resolve_bulk_choice_map_from_matrix(
    matrix_editor_df: pd.DataFrame,
    bulk_results: Dict[str, dict],
) -> Dict[str, str]:
    selected_choice_map: Dict[str, str] = {}
    if matrix_editor_df is None or matrix_editor_df.empty:
        return selected_choice_map

    for _, row in matrix_editor_df.iterrows():
        source_product_id = row.get("source_product_id")
        if source_product_id not in bulk_results:
            continue
        ranking_top = bulk_results[source_product_id]["ranking_top"].copy().reset_index(drop=True)
        if ranking_top.empty:
            continue
        selected_rank = safe_float(row.get("opcion_elegida"))
        if selected_rank is None:
            selected_rank = 1
        selected_rank = int(max(1, min(len(ranking_top), int(round(selected_rank)))))
        candidate_id = ranking_top.iloc[selected_rank - 1]["candidate_id"]
        selected_choice_map[source_product_id] = candidate_id
    return selected_choice_map


def format_bulk_matrix_for_export(matrix_df: pd.DataFrame) -> pd.DataFrame:
    if matrix_df is None or matrix_df.empty:
        return pd.DataFrame()
    export_df = matrix_df.copy()
    export_df = export_df.rename(
        columns={
            "source_product_id": "source_product_id",
            "producto_origen": "producto_origen",
            "precio_origen_eur_t": "precio_origen_eur_t",
            "opcion_elegida": "opcion_elegida",
        }
    )
    numeric_cols = [
        col
        for col in export_df.columns
        if any(token in col for token in ["score_", "precio_", "dif_precio_"])
        or col in {"precio_origen_eur_t", "opcion_elegida"}
    ]
    for col in numeric_cols:
        try:
            export_df[col] = pd.to_numeric(export_df[col], errors="coerce")
        except Exception:
            export_df[col] = export_df[col].apply(safe_float)
    return export_df
def build_bulk_report_text(
    selected_choice_map: Dict[str, str],
    bulk_results: Dict[str, dict],
    summary_df: pd.DataFrame,
    selected_metrics: List[str],
    metric_weights: Dict[str, float],
    component_weights: Dict[str, float],
    source_repo: Repository,
    target_repo: Repository,
) -> str:
    lines: List[str] = []
    lines.append("INFORME DE COMPARATIVA MÚLTIPLE - REBRANDING")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1. CONTEXTO")
    lines.append(f"- Archivo origen: {source_repo.filename}")
    lines.append(f"- Archivo destino: {target_repo.filename}")
    lines.append(f"- Piensos origen analizados: {len(selected_choice_map)}")
    lines.append("")
    lines.append("2. METODOLOGÍA HEURÍSTICA")
    lines.append("- Para cada pienso origen se calcula un ranking independiente frente al portfolio destino filtrado.")
    lines.append("- Un score más bajo indica mayor parecido relativo dentro de la comparación de ese pienso origen.")
    lines.append("- La selección final debe validarse técnicamente antes de migrar la referencia.")
    lines.append("")
    lines.append("Pesos globales del score:")
    for component, value in component_weights.items():
        lines.append(f"  * {component}: {value:.2f}")
    lines.append("")
    lines.append("Métricas seleccionadas y peso específico:")
    for metric in selected_metrics:
        lines.append(f"  * {metric}: {metric_weights.get(metric, 1.0):.2f}")

    if not summary_df.empty:
        lines.append("")
        lines.append("3. SELECCIÓN FINAL POR PIENSO ORIGEN")
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['producto_origen']} -> {row['producto_destino_elegido']} | "
                f"Score total {row['score_total']:.3f} | "
                f"Precio origen {row['precio_origen_eur_t']:.3f} €/t | "
                f"Precio destino {row['precio_destino_eur_t']:.3f} €/t | "
                f"Dif. precio {row['dif_precio_pct']:.2f}%"
            )

    for source_product_id, candidate_id in selected_choice_map.items():
        result = bulk_results.get(source_product_id)
        if not result:
            continue
        source_row = result["source_product_row"]
        ranking_row = result["ranking_df"].set_index("candidate_id").loc[candidate_id]
        metric_details = result["detail_map"][candidate_id]["metric_details"].copy()
        ingredient_details = result["detail_map"][candidate_id]["ingredient_details"].copy()

        lines.append("")
        lines.append("-" * 72)
        lines.append(f"4. DETALLE PARA {source_row['display_name']}")
        lines.append(f"Destino elegido: {ranking_row['display_name']}")
        lines.append(f"Score total: {ranking_row['total_score']:.3f}")
        lines.append(f"Dif. precio %: {ranking_row['price_diff_pct']:.2f}%")
        lines.append("Principales diferencias analíticas:")
        metric_subset = metric_details.sort_values("impact", ascending=False).head(6)
        if metric_subset.empty:
            lines.append("  * Sin detalle analítico disponible.")
        else:
            for _, metric_row in metric_subset.iterrows():
                lines.append(
                    "  * "
                    f"{metric_row['metric']}: origen {display_float(metric_row['actual'], 4)} | "
                    f"destino {display_float(metric_row['candidate'], 4)} | "
                    f"dif. abs. {display_float(metric_row['abs_gap'], 4)} | "
                    f"impacto {display_float(metric_row['impact'], 4)}"
                )
        lines.append("Principales diferencias de ingredientes:")
        ingredient_subset = ingredient_details[ingredient_details["abs_gap_pct"] > 0].sort_values("abs_gap_pct", ascending=False).head(6)
        if ingredient_subset.empty:
            lines.append("  * Sin diferencias relevantes de ingredientes.")
        else:
            for _, ing_row in ingredient_subset.iterrows():
                lines.append(
                    "  * "
                    f"{ing_row['ingredient']}: origen {display_float(ing_row['actual_pct'], 3)}% | "
                    f"destino {display_float(ing_row['candidate_pct'], 3)}% | "
                    f"dif. abs. {display_float(ing_row['abs_gap_pct'], 3)}%"
                )

    lines.append("")
    lines.append("5. CONCLUSIÓN OPERATIVA")
    lines.append("- El análisis múltiple sirve para tratar lotes de SKUs y acelerar la propuesta de equivalencias.")
    lines.append("- Los candidatos elegidos deben revisarse por ajuste nutricional, límites, coste y encaje comercial.")
    return "\n".join(lines)


def build_bulk_excel_export(
    summary_df: pd.DataFrame,
    nutrient_wide_df: pd.DataFrame,
    metric_long_df: pd.DataFrame,
    ingredient_long_df: pd.DataFrame,
    consolidated_ranking_df: pd.DataFrame,
    bulk_report_text: str,
    selection_matrix_df: pd.DataFrame | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if summary_df.empty:
            pd.DataFrame({"mensaje": ["Sin resultados para exportar"]}).to_excel(writer, sheet_name="Resumen", index=False)
        else:
            summary_df.to_excel(writer, sheet_name="Resumen", index=False)
        if selection_matrix_df is not None and not selection_matrix_df.empty:
            selection_matrix_df.to_excel(writer, sheet_name="Matriz_seleccion", index=False)
        if not nutrient_wide_df.empty:
            nutrient_wide_df.to_excel(writer, sheet_name="Comparativa_nutrientes", index=False)
        if not metric_long_df.empty:
            metric_long_df.to_excel(writer, sheet_name="Metricas_largo", index=False)
        if not ingredient_long_df.empty:
            ingredient_long_df.to_excel(writer, sheet_name="Ingredientes_largo", index=False)
        if not consolidated_ranking_df.empty:
            consolidated_ranking_df.to_excel(writer, sheet_name="Ranking_consolidado", index=False)
        pd.DataFrame({"Informe": bulk_report_text.splitlines()}).to_excel(writer, sheet_name="Informe", index=False)

        for ws in writer.book.worksheets:
            for col_idx, column_cells in enumerate(ws.columns, start=1):
                max_len = 0
                for cell in column_cells:
                    try:
                        max_len = max(max_len, len(str(cell.value or "")))
                    except Exception:
                        pass
                ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 10), 40)

    buffer.seek(0)
    return buffer.getvalue()


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
analysis_mode = st.radio(
    "Modo de análisis",
    ["Comparativa individual", "Comparativa múltiple origen"],
    horizontal=True,
    help="La comparativa individual mantiene el flujo actual. La comparativa múltiple permite seleccionar varios piensos del archivo origen y obtener propuestas de destino para todos a la vez.",
)

sel_col1, sel_col2 = st.columns([1, 1])
with sel_col1:
    selected_source_species, source_products_filtered = repository_selector("Especie origen", source_repo)
    source_products_filtered = source_products_filtered.sort_values(["product_name", "spec_id"]).reset_index(drop=True)
    if analysis_mode == "Comparativa individual":
        source_product_id = st.selectbox(
            "Pienso actual",
            source_products_filtered["product_id"].tolist(),
            format_func=lambda pid: source_products_filtered.set_index("product_id").loc[pid, "display_name"],
        )
        selected_source_ids = [source_product_id]
    else:
        selected_source_ids = st.multiselect(
            "Piensos origen (selección múltiple)",
            options=source_products_filtered["product_id"].tolist(),
            default=source_products_filtered["product_id"].tolist()[: min(5, len(source_products_filtered))],
            format_func=lambda pid: source_products_filtered.set_index("product_id").loc[pid, "display_name"],
            help="Puedes seleccionar 20 piensos o más a la vez. La aplicación calculará un ranking independiente para cada uno frente al fichero destino.",
        )
        if len(selected_source_ids) > 20:
            st.caption(f"Selección actual: {len(selected_source_ids)} piensos origen.")

with sel_col2:
    selected_target_species, target_products_filtered = repository_selector(
        "Especie destino", target_repo, default_species=selected_source_species
    )
    compare_all_target_species = st.checkbox("Comparar contra toda la gama destino", value=False)
    if compare_all_target_species:
        target_products_filtered = target_repo.products.copy()
    if analysis_mode == "Comparativa individual":
        source_product_row = source_repo.products.set_index("product_id").loc[selected_source_ids[0]]
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
    else:
        source_info_df = pd.DataFrame(
            {
                "Campo": ["Especie origen", "Piensos origen seleccionados", "Especie destino", "Candidatos destino"],
                "Valor": [
                    selected_source_species,
                    len(selected_source_ids),
                    "Todas" if compare_all_target_species else selected_target_species,
                    len(target_products_filtered),
                ],
            }
        )
    st.write("**Contexto de la ejecución**")
    st.dataframe(source_info_df, use_container_width=True, hide_index=True)

if analysis_mode == "Comparativa múltiple origen" and not selected_source_ids:
    st.warning("Selecciona al menos un pienso origen para ejecutar la comparativa múltiple.")
    st.stop()

candidate_ids = [pid for pid in target_products_filtered["product_id"].tolist() if pid not in set(selected_source_ids)]
target_metric_set = set(target_repo.analyses["analysis_name"].dropna().unique().tolist())
if analysis_mode == "Comparativa individual":
    source_metrics = source_repo.analyses[source_repo.analyses["product_id"] == selected_source_ids[0]][
        ["analysis_name", "level", "min", "max", "limit_type"]
    ].copy()
    common_metrics = sorted(set(source_repo.nutrient_map.get(selected_source_ids[0], {}).keys()) & target_metric_set)
else:
    source_metrics = pd.DataFrame()
    source_metric_union = set()
    for source_id in selected_source_ids:
        source_metric_union |= set(source_repo.nutrient_map.get(source_id, {}).keys())
    common_metrics = sorted(source_metric_union & target_metric_set)

if not common_metrics:
    st.error("No hay métricas analíticas comunes entre la selección origen y la gama destino filtrada.")
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
    if analysis_mode == "Comparativa individual":
        st.caption(
            "La selección de métricas y sus pesos se conserva por especie durante la sesión para no tener que redefinirla en cada comparación."
        )
    else:
        st.caption(
            "La selección de métricas y sus pesos se reutiliza también en la comparativa múltiple. La misma configuración se aplicará a todos los piensos origen seleccionados."
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
    if analysis_mode == "Comparativa individual":
        st.write("**Producto origen**")
    else:
        st.write("**Selección origen**")
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

if analysis_mode == "Comparativa individual":
    source_product_row = source_repo.products.set_index("product_id").loc[selected_source_ids[0]]
    ranking_df, detail_map = compare_product_against_candidates(
        base_product_id=selected_source_ids[0],
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

        export_df = ranking_top[
            [
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
            ]
        ].copy()
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
        st.download_button(
            label="Descargar ranking en CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ranking_rebranding.csv",
            mime="text/csv",
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

        excel_bytes = build_excel_export(
            source_product_row=source_product_row,
            ranking_top=ranking_top,
            detail_map=detail_map,
            report_text=report_text,
        )
        st.download_button(
            label="Descargar resultados en Excel",
            data=excel_bytes,
            file_name=f"comparativa_{source_product_row['spec_id']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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

else:
    bulk_results, consolidated_ranking_df = compare_multiple_source_products(
        source_product_ids=selected_source_ids,
        candidate_ids=candidate_ids,
        source_repo=source_repo,
        target_repo=target_repo,
        selected_metrics=selected_metrics,
        metric_weights=metric_weights,
        component_weights=component_weights,
        top_n=top_n,
    )

    if not bulk_results:
        st.warning("No se han encontrado candidatos destino para la selección múltiple actual.")
        st.stop()

    st.subheader(f"Comparativa múltiple de {len(bulk_results)} piensos origen")
    bulk_tabs = st.tabs(["Selección final", "Rankings por origen", "Detalle elegido", "Datos parseados"])

    selected_choice_map: Dict[str, str] = {}
    with bulk_tabs[0]:
        st.caption(
            "Esta vista permite trabajar rápido con muchos piensos origen a la vez. Cada fila es un pienso origen y el campo 'Opción elegida' indica qué candidato del top se selecciona para ese origen."
        )
        matrix_base_df = build_bulk_matrix_editor_df(
            source_product_ids=selected_source_ids,
            bulk_results=bulk_results,
            top_n=top_n,
        )
        matrix_visible_df = matrix_base_df.drop(columns=["source_product_id"], errors="ignore")
        candidate_option_count = max(
            1,
            max(
                len(bulk_results[source_product_id]["ranking_top"])
                for source_product_id in selected_source_ids
                if source_product_id in bulk_results
            ),
        )
        st.write("**Matriz rápida de selección origen → destino**")
        edited_matrix_visible_df = st.data_editor(
            matrix_visible_df,
            key=f"bulk_matrix_editor_{species_key}_{len(selected_source_ids)}_{top_n}",
            use_container_width=True,
            hide_index=True,
            disabled=[col for col in matrix_visible_df.columns if col != "opcion_elegida"],
            column_config={
                "opcion_elegida": st.column_config.SelectboxColumn(
                    "Opción elegida",
                    help="Selecciona el número de opción del top para cada pienso origen.",
                    options=list(range(1, candidate_option_count + 1)),
                    required=True,
                ),
                "precio_origen_eur_t": st.column_config.NumberColumn("Precio origen €/t", format="%.3f"),
                **{
                    f"score_{i}": st.column_config.NumberColumn(f"Score {i}", format="%.3f")
                    for i in range(1, candidate_option_count + 1)
                    if f"score_{i}" in matrix_visible_df.columns
                },
                **{
                    f"precio_{i}_eur_t": st.column_config.NumberColumn(f"Precio {i} €/t", format="%.3f")
                    for i in range(1, candidate_option_count + 1)
                    if f"precio_{i}_eur_t" in matrix_visible_df.columns
                },
                **{
                    f"dif_precio_{i}_pct": st.column_config.NumberColumn(f"Dif. precio {i} %", format="%.2f")
                    for i in range(1, candidate_option_count + 1)
                    if f"dif_precio_{i}_pct" in matrix_visible_df.columns
                },
            },
        )
        matrix_editor_df = edited_matrix_visible_df.copy()
        matrix_editor_df.insert(0, "source_product_id", matrix_base_df["source_product_id"].values)
        selected_choice_map = resolve_bulk_choice_map_from_matrix(matrix_editor_df, bulk_results)

        summary_df, nutrient_wide_df, metric_long_df, ingredient_long_df = build_bulk_selection_tables(
            selected_choice_map=selected_choice_map,
            bulk_results=bulk_results,
            source_repo=source_repo,
            target_repo=target_repo,
            selected_metrics=selected_metrics,
        )
        if not summary_df.empty:
            st.write("**Selección final consolidada**")
            st.dataframe(summary_df.round(3), use_container_width=True, hide_index=True)
        if not nutrient_wide_df.empty:
            st.write("**Comparativa de precios y nutrientes elegidos**")
            st.dataframe(nutrient_wide_df.round(4), use_container_width=True, hide_index=True)

        bulk_report_text = build_bulk_report_text(
            selected_choice_map=selected_choice_map,
            bulk_results=bulk_results,
            summary_df=summary_df,
            selected_metrics=selected_metrics,
            metric_weights=metric_weights,
            component_weights=component_weights,
            source_repo=source_repo,
            target_repo=target_repo,
        )
        bulk_excel_bytes = build_bulk_excel_export(
            summary_df=summary_df,
            nutrient_wide_df=nutrient_wide_df,
            metric_long_df=metric_long_df,
            ingredient_long_df=ingredient_long_df,
            consolidated_ranking_df=consolidated_ranking_df,
            bulk_report_text=bulk_report_text,
            selection_matrix_df=format_bulk_matrix_for_export(matrix_editor_df),
        )
        st.download_button(
            label="Descargar selección múltiple en Excel",
            data=bulk_excel_bytes,
            file_name="comparativa_multiple_rebranding.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with bulk_tabs[1]:
        st.caption("Cada bloque muestra el top de candidatos para uno de los piensos origen seleccionados.")
        for source_product_id in selected_source_ids:
            if source_product_id not in bulk_results:
                continue
            source_row = bulk_results[source_product_id]["source_product_row"]
            ranking_top = bulk_results[source_product_id]["ranking_top"]
            with st.expander(f"{source_row['display_name']} - Top {len(ranking_top)}", expanded=False):
                st.dataframe(format_ranking_for_display(ranking_top), use_container_width=True, hide_index=True)

    with bulk_tabs[2]:
        detail_source_id = st.selectbox(
            "Pienso origen para revisar el detalle del destino elegido",
            [pid for pid in selected_source_ids if pid in bulk_results],
            format_func=lambda pid: bulk_results[pid]["source_product_row"]["display_name"],
        )
        chosen_candidate_id = selected_choice_map.get(detail_source_id)
        if chosen_candidate_id is None:
            chosen_candidate_id = bulk_results[detail_source_id]["ranking_top"]["candidate_id"].iloc[0]
        st.write(
            f"**Origen:** {bulk_results[detail_source_id]['source_product_row']['display_name']}  \\n**Destino elegido:** {bulk_results[detail_source_id]['ranking_df'].set_index('candidate_id').loc[chosen_candidate_id, 'display_name']}"
        )
        render_candidate_detail(chosen_candidate_id, bulk_results[detail_source_id]["ranking_df"], bulk_results[detail_source_id]["detail_map"])

    with bulk_tabs[3]:
        render_repository_debug(source_repo, "Depuración archivo origen")
        render_repository_debug(target_repo, "Depuración archivo destino")
        with st.expander("Ranking consolidado", expanded=False):
            st.dataframe(consolidated_ranking_df.round(3), use_container_width=True, hide_index=True)
