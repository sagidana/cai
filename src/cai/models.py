"""models: the model catalogue and pin store, persisted at one file.

ModelsRegistry owns ~/.config/cai/models.json, a single JSON object holding:
  - fetched_at: unix time the catalogue was last pulled from the provider
  - models:     the cached catalogue (records from OpenAiApi.get_models)
  - pinned:     the model ids the user starred in the :models picker

the catalogue is fetched lazily from the provider's /models endpoint and cached
to disk, refreshed once it is older than a day; offline or on failure it falls
back to whatever is cached (even if stale). pins are a UI preference and never
expire. every operation reads and writes the file fresh, so separate registry
instances (the picker's, and the overlay's pin toggles) stay consistent.

load_favorites / toggle_favorite are the module-level adapters the model overlay
calls; they route the overlay's star toggles through the same file."""
import os
import json
import time
import logging

from cai import config


log = logging.getLogger("cai")

# refresh the cached catalogue when it is older than this many seconds.
_CACHE_MAX_AGE = 86400


def models_path():
    return os.path.join(config.config_dir(), "models.json")


class ModelsRegistry:
    """read/write access to the cached model catalogue and the pinned set."""

    def __init__(self, api=None, path=None, max_age=_CACHE_MAX_AGE):
        # api is only needed to (re)fetch the catalogue; pin reads/writes never
        # touch it, so the overlay can use an api-less registry.
        self.api = api
        self.path = path or models_path()
        self.max_age = max_age

    def _read(self):
        """the file as a dict, or {} when missing or unreadable."""
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write(self, data):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("could not write models cache %s: %s", self.path, e)

    def _is_fresh(self, data):
        if "models" not in data:
            return False
        age = time.time() - data.get("fetched_at", 0)
        return age < self.max_age

    def models(self, refresh=False):
        """the cached catalogue (list of records). fetches from the provider and
        rewrites the cache when missing/stale (or refresh=True); on a fetch
        failure it keeps whatever is cached."""
        data = self._read()
        if not refresh and self._is_fresh(data):
            return data.get("models", [])

        fetched = None
        if self.api is not None:
            fetched = self.api.get_models()
        if not fetched:
            if data.get("models"):
                log.warning("models: fetch failed, using cached catalogue (%d)",
                            len(data["models"]))
            return data.get("models", [])

        data["models"] = fetched
        data["fetched_at"] = int(time.time())
        self._write(data)
        log.info("models: fetched %d from provider", len(fetched))
        return fetched

    def model_ids(self, refresh=False):
        ids = []
        for record in self.models(refresh=refresh):
            ids.append(record["id"])
        return ids

    def prices(self, refresh=False):
        """{model_id: record} for the picker's price labels."""
        prices = {}
        for record in self.models(refresh=refresh):
            prices[record["id"]] = record
        return prices

    def context_length(self, model_id):
        """the cached context window for model_id, or None when not cached.

        cache-only (never fetches), so callers on a hot path - e.g. the status
        line's ctx readout - can resolve it cheaply without blocking on the
        network. populated by models()/get_models, which store context_length
        per record."""
        for record in self._read().get("models", []):
            if record.get("id") == model_id:
                return record.get("context_length")
        return None

    def pinned(self):
        """the set of pinned model ids."""
        data = self._read()
        ids = set()
        for item in data.get("pinned", []):
            if isinstance(item, str):
                ids.add(item)
        return ids

    def toggle_pin(self, model_id):
        """flip model_id in or out of the pinned set; returns the new set
        (sorted on disk for a stable file)."""
        pinned = self.pinned()
        if model_id in pinned:
            pinned.discard(model_id)
        else:
            pinned.add(model_id)
        data = self._read()
        data["pinned"] = sorted(pinned)
        self._write(data)
        return pinned


# --- module-level adapters used by the model overlay (overlays/model.py) ---

def load_favorites():
    """the pinned model ids, for the overlay's favorites section."""
    return ModelsRegistry().pinned()


def toggle_favorite(model_id):
    """flip one model's pin - the overlay's Tab toggle."""
    return ModelsRegistry().toggle_pin(model_id)
