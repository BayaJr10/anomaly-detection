"""Reverse-geocode sensor locations to neighbourhood names.

The published dataset was geocoded with OpenStreetMap Nominatim. Nominatim answers change
as OSM is edited, so a cache (data/reference/sensor_locations.csv) freezes the names used in
the published dataset. Cached locations are never re-queried unless their coordinates
change; only new locations go to the network. This makes the pipeline reproducible offline.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from airq.config import load_config, resolve

log = logging.getLogger(__name__)

# Nominatim address keys tried in order: the first one present becomes the neighbourhood name.
ADDRESS_PRIORITY = [
    "suburb", "neighbourhood", "quarter", "city_district", "district",
    "village", "town", "county", "city", "state_district",
]
CACHE_COLUMNS = ["LOCATION", "LAT", "LON", "NEIGHBORHOOD"]


def load_cache(path=None) -> pd.DataFrame:
    path = resolve(path or load_config()["ingest"]["geocode_cache"])
    if not path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    return pd.read_csv(path)


def save_cache(cache: pd.DataFrame, path=None) -> None:
    path = resolve(path or load_config()["ingest"]["geocode_cache"])
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.sort_values("LOCATION")[CACHE_COLUMNS].to_csv(path, index=False)


def pick_name(address: dict) -> str | None:
    for key in ADDRESS_PRIORITY:
        if address.get(key):
            return address[key]
    return None


def make_reverse(user_agent: str):
    """Rate-limited Nominatim reverse geocoder (1 request/second, per the usage policy)."""
    from geopy.extra.rate_limiter import RateLimiter
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent=user_agent, timeout=30)
    return RateLimiter(geolocator.reverse, min_delay_seconds=1.1, max_retries=3)


def geocode_locations(coords: pd.DataFrame, cfg: dict | None = None,
                      allow_network: bool = True, tol: float = 1e-4) -> pd.DataFrame:
    """coords: LOCATION, LAT, LON  ->  LOCATION, NEIGHBORHOOD (and LAT, LON passed through)."""
    cfg = cfg or load_config()
    no_coords = cfg["ingest"]["no_coordinates_label"]
    cache = load_cache().set_index("LOCATION")
    reverse = None
    names, new_rows = [], []
    for _, r in coords.iterrows():
        loc, lat, lon = int(r["LOCATION"]), r["LAT"], r["LON"]
        if pd.isna(lat) or pd.isna(lon):
            names.append(no_coords)
            continue
        if loc in cache.index:
            c = cache.loc[loc]
            same = (not pd.isna(c["LAT"]) and abs(c["LAT"] - lat) < tol and abs(c["LON"] - lon) < tol)
            if same:
                names.append(c["NEIGHBORHOOD"])
                continue
            log.warning("Location %s moved (cache %.4f,%.4f vs %.4f,%.4f); re-geocoding",
                        loc, c["LAT"], c["LON"], lat, lon)
        if not allow_network:
            log.warning("Location %s not in cache and network geocoding disabled -> 'Unknown'", loc)
            names.append("Unknown")
            continue
        if reverse is None:
            reverse = make_reverse(cfg["ingest"]["geocode_user_agent"])
        try:
            res = reverse((float(lat), float(lon)), zoom=14, language="en", addressdetails=True)
            name = pick_name(res.raw.get("address", {})) if res else None
        except Exception as e:  # network or service error: keep going, mark as unknown
            log.warning("Geocoding failed for %s: %s", loc, e)
            name = None
        name = name or "Unknown"
        names.append(name)
        new_rows.append({"LOCATION": loc, "LAT": lat, "LON": lon, "NEIGHBORHOOD": name})
    if new_rows:
        updated = pd.concat([cache.reset_index(), pd.DataFrame(new_rows)]).drop_duplicates("LOCATION", keep="last")
        save_cache(updated)
    out = coords.copy()
    out["NEIGHBORHOOD"] = names
    return out


def cache_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze the location -> neighbourhood mapping of an existing hourly dataset."""
    c = (df.groupby("LOCATION")
           .agg(LAT=("NEIGHBORHOOD_LAT", "first"), LON=("NEIGHBORHOOD_LON", "first"),
                NEIGHBORHOOD=("NEIGHBORHOOD", "first"))
           .reset_index())
    return c[CACHE_COLUMNS]
