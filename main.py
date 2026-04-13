from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
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


def split_formula_blocks(lines: List[str]) -> List[List[str]]:
    starts = [idx for idx, line in enumerate(lines) if "Specification:" in line]
    if not starts:
        raise ValueError(
            "No se han encontrado bloques con 'Specification:'. El Excel no parece tener el mismo formato que el ejemplo."
        )

    blocks: List[List[str]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else len(lines) - 1
        block = lines[start : end + 1]
        blocks.append(block)
    return blocks


def parse_spec_line(line: str) -> dict:
    match = SPEC_LINE_RE.search(line)
    if not match:
        return {}

    raw_product = normalize_text(match.group(2))
    code, product_name = split_product_code_and_name(raw_product)
    return {
        "spec_id": match.group(1),
        "product_raw": raw_product,
        "product_code": code,
        "product_name": product_name,
        "cost_tonne": safe_float(match.group(3)),
    }


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
    }
    for idx, line in enumerate(block_lines):
        if "INCLUDED RAW MATERIALS" in line:
            positions["ingredients_start"] = idx + 2
        elif line.strip().startswith("ANALYSIS"):
            positions["analysis_start"] = idx + 2
        elif "RAW MATERIAL SENSITIVITY" in line:
            positions["sensitivity_start"] = idx
            break
    return positions


def parse_ingredient_line(line: str) -> dict | None:
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


def parse_analysis_line(line: str) -> dict | None:
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
            }
        )

        if positions["ingredients_start"] is not None and positions["analysis_start"] is not None:
            ingredient_lines = block[positions["ingredients_start"] : positions["analysis_start"] - 2]
            for line in ingredient_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("-"):
                    continue
                parsed = parse_ingredient_line(line)
                if parsed:
                    ingredient_rows.append({"product_id": product_id, **parsed})

        if positions["analysis_start"] is not None:
            end = positions["sensitivity_start"] if positions["sensitivity_start"] is not None else len(block)
            analysis_lines = block[positions["analysis_start"] : end]
            for line in analysis_lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("-") or "RAW MATERIAL SENSITIVITY" in stripped:
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


def repository_selector(label: str, repository: Repository, default_species: str | None = None) -> Tuple[str, pd.DataFrame]:
    species_values = sorted(repository.products["species"].dropna().unique().tolist())
    if not species_values:
        species_values = ["Sin clasificar"]

    species_options = ["Todas"] + species_values
    default_index = species_options.index(default_species) if default_species in species_options else 0
    selected_species = st.selectbox(label, species_options, index=default_index)

    if selected_species == "Todas":
        filtered_products = repository.products.copy()
    else:
        filtered_products = repository.products[repository.products["species"] == selected_species].copy()

    filtered_products = filtered_products.sort_values(["species", "product_name", "spec_id"]).reset_index(drop=True)
    return selected_species, filtered_products


st.title("Comparador técnico de fórmulas | Rebranding")
st.caption(
    "La aplicación lee el formato de exportación tipo Multi-Mix del Excel, reconstruye cada fórmula y propone los productos destino más parecidos."
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
    st.info(
        "El score es heurístico: combina diferencias de nutrientes, composición de ingredientes, compatibilidad de límites y precio. Un score más bajo indica mayor similitud."
    )

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
        "Especie destino", target_repo, default_species=selected_source_species if selected_source_species != "Todas" else None
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

default_metrics = auto_select_metrics(common_metrics, selected_source_species)

st.subheader("Selección de métricas críticas")
metric_selector_col1, metric_selector_col2 = st.columns([1.4, 1])
with metric_selector_col1:
    selected_metrics = st.multiselect(
        "Métricas incluidas en la comparación",
        options=common_metrics,
        default=default_metrics,
        help="Puedes ajustar manualmente qué nutrientes/analíticas entran en el score.",
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
        "Peso": [default_metric_weight(metric, selected_source_species) for metric in selected_metrics],
    }
)

st.write("**Pesos por métrica**")
edited_weights_df = st.data_editor(
    weight_editor_df,
    use_container_width=True,
    hide_index=True,
    disabled=["Métrica"],
    column_config={"Peso": st.column_config.NumberColumn(min_value=0.1, max_value=10.0, step=0.1)},
)
metric_weights = dict(zip(edited_weights_df["Métrica"], edited_weights_df["Peso"]))

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
    st.download_button(
        label="Descargar ranking en CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ranking_rebranding.csv",
        mime="text/csv",
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
