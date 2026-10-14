from fastapi import APIRouter, Header, HTTPException

from src.api.schemas import CatalogUpdateRequest, CatalogUpdateResponse
from src.cache.invalidation import publish_catalog_update
from src.config import settings
from src.ingestion.ingest import reingest_item

router = APIRouter()


def _check_admin_token(x_admin_token: str | None) -> None:
    # Internal-only auth for this phase — see docs/threat-model.md threat #8.
    # Not a substitute for real authn/authz before any public deployment.
    if not x_admin_token or x_admin_token != settings.catalog_admin_token:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")


@router.post("/v1/catalog/items", response_model=CatalogUpdateResponse)
def update_item(req: CatalogUpdateRequest, x_admin_token: str | None = Header(default=None)) -> CatalogUpdateResponse:
    _check_admin_token(x_admin_token)
    item = req.item
    if "id" not in item:
        raise HTTPException(status_code=400, detail="item must include an 'id' field")

    affected_chunk_ids = reingest_item(item)
    evicted = publish_catalog_update([item["id"]])

    return CatalogUpdateResponse(item_id=item["id"], affected_chunk_ids=affected_chunk_ids, cache_entries_evicted=evicted)
