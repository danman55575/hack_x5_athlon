from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_raw


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_mean",
]
SETTLEMENT_SUFFIXES = {
    "г",
    "г.",
    "пгт",
    "пгт.",
    "рп",
    "рп.",
    "пос",
    "пос.",
    "п",
    "п.",
    "с",
    "с.",
    "д",
    "д.",
    "ст",
    "ст.",
    "ст-ца",
    "станица",
    "х",
    "х.",
    "хутор",
    "аул",
}
REGION_STOPWORDS = {
    "обл",
    "область",
    "край",
    "респ",
    "республика",
    "автономный",
    "автономная",
    "автономное",
    "округ",
    "г",
    "г.",
}


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.lower().replace("ё", "е").replace("-", " ")
    for token in [".", ",", "(", ")", '"', "'"]:
        text = text.replace(token, " ")
    return " ".join(text.split())


def _normalize_locality(value: Any) -> str:
    tokens = _normalize_text(value).split()
    while tokens and tokens[-1] in SETTLEMENT_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _normalize_region(value: Any) -> str:
    tokens = [token for token in _normalize_text(value).split() if token not in REGION_STOPWORDS]
    return " ".join(tokens)


def _fetch_json(url: str, pause_s: float, retries: int = 3) -> dict[str, Any] | list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if pause_s > 0:
                time.sleep(pause_s)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait_s = pause_s * attempt if pause_s > 0 else attempt
            print(f"Повтор запроса через {wait_s:.1f}с после ошибки: {exc}")
            time.sleep(wait_s)
    assert last_error is not None
    raise last_error


def _candidate_queries(locality: str, region: str, geo_level: str) -> list[str]:
    if geo_level == "region":
        region_norm = _normalize_region(region)
        return [region, region_norm]

    locality_norm = _normalize_locality(locality)
    region_norm = _normalize_region(region)
    queries = [
        locality_norm,
        str(locality),
        f"{locality_norm} {region_norm}",
    ]
    return [query for query in queries if query]


def _score_candidate(candidate: dict[str, Any], locality: str, region: str, geo_level: str) -> float:
    if geo_level == "region":
        target_region = _normalize_region(region)
        admin = " ".join(
            _normalize_text(candidate.get(key, ""))
            for key in ("name", "admin1", "admin2", "admin3")
        )
        return float(
            10 * (target_region == _normalize_region(candidate.get("name", "")))
            + 6 * (target_region == _normalize_region(candidate.get("admin1", "")))
            + 3 * (target_region in admin)
            + math.log1p(float(candidate.get("population") or 0.0)) / 10.0
        )

    target_locality = _normalize_locality(locality)
    target_region = _normalize_region(region)
    cand_name = _normalize_locality(candidate.get("name", ""))
    cand_admin1 = _normalize_region(candidate.get("admin1", ""))
    cand_admin2 = _normalize_region(candidate.get("admin2", ""))
    admin_blob = " ".join([cand_admin1, cand_admin2, _normalize_region(candidate.get("admin3", ""))]).strip()

    score = 0.0
    if cand_name == target_locality:
        score += 8.0
    elif cand_name.startswith(target_locality) or target_locality.startswith(cand_name):
        score += 5.0
    elif target_locality in cand_name:
        score += 3.0

    if target_region == cand_admin1:
        score += 6.0
    elif target_region == cand_admin2:
        score += 4.0
    elif target_region and target_region in admin_blob:
        score += 2.0

    score += math.log1p(float(candidate.get("population") or 0.0)) / 10.0
    return float(score)


def _resolve_one(query_row: pd.Series, geo_level: str, pause_s: float) -> dict[str, Any]:
    locality = query_row.get("locality", "")
    region = query_row["region"]
    queries = _candidate_queries(locality, region, geo_level)

    best_candidate: dict[str, Any] | None = None
    best_score = float("-inf")
    best_query = None
    for query in queries:
        url = f"{GEOCODING_URL}?{urlencode({'name': query, 'count': 10, 'language': 'ru', 'countryCode': 'RU'})}"
        payload = _fetch_json(url, pause_s=pause_s)
        candidates = payload.get("results", []) if isinstance(payload, dict) else []
        for candidate in candidates:
            score = _score_candidate(candidate, locality, region, geo_level)
            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_query = query

    result = query_row.to_dict()
    result["query_used"] = best_query
    result["score"] = None if best_candidate is None else float(best_score)
    if best_candidate is None:
        result["latitude"] = np.nan
        result["longitude"] = np.nan
        result["resolved_name"] = None
        result["admin1"] = None
        result["admin2"] = None
        result["country"] = None
        result["geocode_status"] = "not_found"
    else:
        result["latitude"] = float(best_candidate["latitude"])
        result["longitude"] = float(best_candidate["longitude"])
        result["resolved_name"] = best_candidate.get("name")
        result["admin1"] = best_candidate.get("admin1")
        result["admin2"] = best_candidate.get("admin2")
        result["country"] = best_candidate.get("country")
        result["geocode_status"] = "ok"
    return result


def _resolve_coordinates(geo_df: pd.DataFrame, coords_path: Path, geo_level: str, pause_s: float) -> pd.DataFrame:
    existing = pd.DataFrame(columns=geo_df.columns.tolist() + [
        "query_used",
        "score",
        "latitude",
        "longitude",
        "resolved_name",
        "admin1",
        "admin2",
        "country",
        "geocode_status",
    ])
    if coords_path.exists():
        existing = pd.read_csv(coords_path)

    key_cols = ["region"] if geo_level == "region" else ["locality", "region"]
    resolved_keys = set(
        tuple(row[col] for col in key_cols)
        for _, row in existing.loc[existing["geocode_status"] == "ok", key_cols].drop_duplicates().iterrows()
    )
    todo = geo_df[
        ~geo_df[key_cols].apply(lambda row: tuple(row[col] for col in key_cols) in resolved_keys, axis=1)
    ].reset_index(drop=True)

    if todo.empty:
        print(f"Координаты уже готовы: {coords_path}")
        return existing

    rows: list[dict[str, Any]] = []
    for idx, row in todo.iterrows():
        rows.append(_resolve_one(row, geo_level=geo_level, pause_s=pause_s))
        if (idx + 1) % 50 == 0:
            print(f"Геокодирование: обработано {idx + 1} / {len(todo)}")

    resolved = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    resolved = resolved.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols).reset_index(drop=True)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_csv(coords_path, index=False, encoding="utf-8")
    print(f"Сохранены координаты: {coords_path}")
    return resolved


def _to_monthly_weather(one_location: dict[str, Any], key_payload: dict[str, Any]) -> pd.DataFrame:
    daily = one_location.get("daily") or {}
    if not daily or "time" not in daily:
        return pd.DataFrame()

    daily_df = pd.DataFrame(daily)
    if daily_df.empty:
        return pd.DataFrame()
    daily_df["date"] = pd.to_datetime(daily_df["time"])
    daily_df["year"] = daily_df["date"].dt.year.astype(np.int16)
    daily_df["month"] = daily_df["date"].dt.month.astype(np.int8)

    temp_mean = daily_df["temperature_2m_mean"].astype(float)
    temp_min = daily_df["temperature_2m_min"].astype(float)
    precip = daily_df["precipitation_sum"].astype(float)
    snowfall = daily_df["snowfall_sum"].astype(float)

    daily_df["freeze_day"] = (temp_min < 0).astype(np.int16)
    daily_df["rain_day"] = (precip > 0.1).astype(np.int16)
    daily_df["snow_day"] = (snowfall > 0.1).astype(np.int16)
    daily_df["hdd"] = np.maximum(18.0 - temp_mean, 0.0)
    daily_df["cdd"] = np.maximum(temp_mean - 22.0, 0.0)

    monthly = (
        daily_df.groupby(["year", "month"], as_index=False)
        .agg(
            wx_temp_mean_obs=("temperature_2m_mean", "mean"),
            wx_temp_max_obs=("temperature_2m_max", "max"),
            wx_temp_min_obs=("temperature_2m_min", "min"),
            wx_precip_sum_obs=("precipitation_sum", "sum"),
            wx_rain_sum_obs=("rain_sum", "sum"),
            wx_snowfall_sum_obs=("snowfall_sum", "sum"),
            wx_precip_hours_obs=("precipitation_hours", "sum"),
            wx_wind_speed_mean_obs=("wind_speed_10m_mean", "mean"),
            wx_freeze_days_obs=("freeze_day", "sum"),
            wx_rain_days_obs=("rain_day", "sum"),
            wx_snow_days_obs=("snow_day", "sum"),
            wx_hdd_obs=("hdd", "sum"),
            wx_cdd_obs=("cdd", "sum"),
        )
    )
    for key, value in key_payload.items():
        monthly[key] = value
    return monthly


def _fetch_weather_monthly(
    coords_df: pd.DataFrame,
    geo_level: str,
    start_date: str,
    end_date: str,
    batch_size: int,
    pause_s: float,
) -> pd.DataFrame:
    ok_coords = coords_df.loc[coords_df["geocode_status"] == "ok"].copy().reset_index(drop=True)
    if ok_coords.empty:
        raise RuntimeError("Нет ни одной успешно геокодированной точки для погоды.")

    rows: list[pd.DataFrame] = []
    for start in range(0, len(ok_coords), batch_size):
        batch = ok_coords.iloc[start:start + batch_size].reset_index(drop=True)
        params = {
            "latitude": ",".join(batch["latitude"].map(lambda x: f"{float(x):.5f}")),
            "longitude": ",".join(batch["longitude"].map(lambda x: f"{float(x):.5f}")),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto",
        }
        url = f"{ARCHIVE_URL}?{urlencode(params)}"
        payload = _fetch_json(url, pause_s=pause_s)
        payload_list = payload if isinstance(payload, list) else [payload]
        if len(payload_list) != len(batch):
            raise RuntimeError(
                f"Ожидалось {len(batch)} погодных ответов в батче, получили {len(payload_list)}."
            )

        for idx, item in enumerate(payload_list):
            key_payload = {"region": batch.iloc[idx]["region"]}
            if geo_level == "locality":
                key_payload["locality"] = batch.iloc[idx]["locality"]
            monthly = _to_monthly_weather(item, key_payload=key_payload)
            if not monthly.empty:
                rows.append(monthly)
        print(f"Погода: обработан батч {start // batch_size + 1}, строк с погодой накоплено {sum(len(x) for x in rows)}")

    weather = pd.concat(rows, ignore_index=True)
    key_cols = ["region"] if geo_level == "region" else ["locality", "region"]
    weather = weather.sort_values(key_cols + ["year", "month"]).reset_index(drop=True)
    return weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/v2.parquet")
    parser.add_argument("--geo-level", choices=["region", "locality"], default="region")
    parser.add_argument("--coords-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    args = parser.parse_args()

    df = load_raw(args.train)
    if args.geo_level == "region":
        geo_df = df[["region"]].drop_duplicates().sort_values(["region"]).reset_index(drop=True)
    else:
        geo_df = df[["locality", "region"]].drop_duplicates().sort_values(["region", "locality"]).reset_index(drop=True)

    coords_path = Path(args.coords_path or f"data/processed/{args.geo_level}_coordinates.csv")
    weather_path = Path(args.weather_path or f"data/processed/weather_monthly_{args.geo_level}.parquet")

    month_min = pd.Timestamp(year=int(df["year"].min()), month=int(df["month"].min()), day=1)
    month_max = pd.Timestamp(year=int(df["year"].max()), month=int(df["month"].max()), day=1) + pd.offsets.MonthEnd(1)
    start_date = args.start_date or month_min.strftime("%Y-%m-%d")
    end_date = args.end_date or month_max.strftime("%Y-%m-%d")

    print(f"Готовим weather dataset: geo_level={args.geo_level}, start={start_date}, end={end_date}")
    coords_df = _resolve_coordinates(geo_df, coords_path=coords_path, geo_level=args.geo_level, pause_s=args.pause_seconds)
    weather = _fetch_weather_monthly(
        coords_df=coords_df,
        geo_level=args.geo_level,
        start_date=start_date,
        end_date=end_date,
        batch_size=int(args.batch_size),
        pause_s=args.pause_seconds,
    )
    weather_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(weather_path, index=False)
    print(f"Сохранён monthly weather dataset: {weather_path}  shape={weather.shape}")


if __name__ == "__main__":
    main()
