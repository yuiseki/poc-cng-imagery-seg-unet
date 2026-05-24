# tests/fixtures

Snapshots of upstream API responses, captured once and committed so
unit tests can run offline + deterministically.

| file | source | command used to capture |
| --- | --- | --- |
| `hotosm_stac_atami.json` | HOTOSM STAC `/search` bbox query around Atami | `curl -X POST https://api.imagery.hotosm.org/stac/search -d '{"bbox":[139.07,35.10,139.10,35.13],"limit":3}'` |
| `hotosm_stac_item_atami.json` | HOTOSM STAC `/search` by id | `curl -X POST https://api.imagery.hotosm.org/stac/search -d '{"ids":["60e5afbe5bc2dc00058bbe06"],"limit":1}'` |
| `overpass_buildings_atami.json` | `overpass.yuiseki.net` `way["building"]` in Atami AOI | `curl -X POST https://overpass.yuiseki.net/api/interpreter --data-urlencode 'data=[out:json][timeout:25];way["building"](35.115,139.075,35.120,139.080);out geom;'` |
| `mpc_sentinel2_atami.json` | Microsoft Planetary Computer Sentinel-2 L2A search | `curl 'https://planetarycomputer.microsoft.com/api/stac/v1/search?collections=sentinel-2-l2a&bbox=139.07,35.10,139.10,35.13&limit=2&datetime=2024-09-01/2024-09-30'` |

E2E tests under `@pytest.mark.integration` still hit the live APIs
(HOTOSM STAC, `overpass.yuiseki.net`, `tile.yuiseki.net`); the
self-hosted `*.yuiseki.net` services have no rate limits worth
worrying about, so they're fair game to call from CI.
