"""Turns one catalog item (see top-level `products.py`) into several
retrievable chunks: an overview, care instructions, policy, and one chunk
per FAQ entry. Each chunk carries enough metadata for the generation step to
describe the item without re-deriving anything from text."""

from src.domain.models import Chunk


def _base_metadata(item: dict) -> dict:
    return {
        "item_id": item["id"],
        "name": item["name"],
        "category": item["category"],
        "fabric": item["fabric"],
        "occasion": item["occasion"],
        "colors": item["colors"],
        "sizes": item["sizes"],
        "price": item["price"],
        "stock_status": item["stock_status"],
    }


def chunk_item(item: dict) -> list[Chunk]:
    meta = _base_metadata(item)
    chunks: list[Chunk] = []

    overview_text = (
        f"Item code {item['id']}. {item['name']} ({item['category']}). {item['description']} "
        f"Fabric: {item['fabric']}. Occasion: {item['occasion']}. "
        f"Colors: {', '.join(item['colors'])}. Sizes: {', '.join(item['sizes'])}. "
        f"Price: {item['price']}. Stock: {item['stock_status']}."
    )
    chunks.append(Chunk(id=f"{item['id']}:overview", item_id=item["id"], chunk_type="overview", text=overview_text, metadata=meta))

    care_text = f"Item code {item['id']}. {item['care_instructions']} {item['return_policy']} {item['exchange_policy']}"
    chunks.append(Chunk(id=f"{item['id']}:care", item_id=item["id"], chunk_type="care", text=care_text, metadata=meta))

    for i, faq in enumerate(item.get("faqs", [])):
        faq_text = f"Q: {faq['question']} A: {faq['answer']}"
        chunks.append(
            Chunk(id=f"{item['id']}:faq:{i}", item_id=item["id"], chunk_type="faq", text=faq_text, metadata=meta)
        )

    return chunks


def chunk_catalog(items: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for item in items:
        out.extend(chunk_item(item))
    return out
