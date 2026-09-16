import logging
from typing import List, Dict, Any, Optional
from app.services.ingestion.sec_parser import SECFilingMetadata, ParsedSection, ParsedBlock

logger = logging.getLogger("finance_agent.ingestion.chunker")


class ChunkItem:
    """Represents a finalized chunk ready for embedding and storage."""
    def __init__(
        self,
        part: Optional[str],
        item: Optional[str],
        sub_section: Optional[str],
        breadcrumb: str,
        chunk_type: str,  # 'narrative' or 'table'
        chunk_index: int,
        content: str,             # What the LLM reads (High fidelity markdown / prose)
        embedding_input: str,     # What the vector model embeds (Search Face + context)
        metadata: Dict[str, Any],
    ):
        self.part = part
        self.item = item
        self.sub_section = sub_section
        self.breadcrumb = breadcrumb
        self.chunk_type = chunk_type
        self.chunk_index = chunk_index
        self.content = content
        self.embedding_input = embedding_input
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "part": self.part,
            "item": self.item,
            "sub_section": self.sub_section,
            "breadcrumb": self.breadcrumb,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding_input": self.embedding_input,
            "metadata": self.metadata,
        }


class SECChunker:
    """Metadata-based chunker for SEC 10-K and 10-Q filings."""

    def __init__(
        self,
        target_chunk_chars: int = 1800,  # ~450 tokens
        min_chunk_chars: int = 200,
        chunk_overlap_chars: int = 200,
    ):
        self.target_chunk_chars = target_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.chunk_overlap_chars = chunk_overlap_chars

    def build_breadcrumb(self, metadata: SECFilingMetadata, section: ParsedSection) -> str:
        parts = [
            f"{metadata.company_name} ({metadata.ticker})",
            f"{metadata.filing_type} FY{metadata.fiscal_year}",
        ]
        if section.part:
            parts.append(section.part)
        if section.item:
            parts.append(f"{section.item}: {section.title}")
        elif section.title:
            parts.append(section.title)
        return " > ".join(parts)

    def chunk_document(
        self,
        metadata: SECFilingMetadata,
        sections: List[ParsedSection],
    ) -> List[ChunkItem]:
        """Converts parsed sections and blocks into discrete, context-enriched ChunkItems."""
        chunks: List[ChunkItem] = []
        global_index = 0

        for section in sections:
            breadcrumb = self.build_breadcrumb(metadata, section)

            # Separate and process blocks in section
            current_text_buffer: List[str] = []
            current_buffer_len = 0

            def flush_text_buffer():
                nonlocal current_text_buffer, current_buffer_len, global_index
                if not current_text_buffer:
                    return

                combined_text = "\n\n".join(current_text_buffer).strip()
                if len(combined_text) >= self.min_chunk_chars:
                    # Search Face for narrative includes breadcrumbs + text
                    embed_text = f"[{breadcrumb}]\n{combined_text}"
                    chunk_meta = {
                        "ticker": metadata.ticker,
                        "company": metadata.company_name,
                        "filing": metadata.filing_type,
                        "fiscal_year": metadata.fiscal_year,
                        "part": section.part,
                        "item": section.item,
                        "section_title": section.title,
                        "chunk_type": "narrative",
                    }
                    chunks.append(
                        ChunkItem(
                            part=section.part,
                            item=section.item,
                            sub_section=section.title,
                            breadcrumb=breadcrumb,
                            chunk_type="narrative",
                            chunk_index=global_index,
                            content=combined_text,
                            embedding_input=embed_text,
                            metadata=chunk_meta,
                        )
                    )
                    global_index += 1

                current_text_buffer = []
                current_buffer_len = 0

            for block in section.blocks:
                if block.block_type == "table":
                    # Flush any preceding narrative before introducing a table
                    flush_text_buffer()

                    # Process table block
                    table_summary = block.summary or block.table_title or "Financial Table"
                    embed_text = f"[{breadcrumb}]\n{table_summary}"

                    table_meta = {
                        "ticker": metadata.ticker,
                        "company": metadata.company_name,
                        "filing": metadata.filing_type,
                        "fiscal_year": metadata.fiscal_year,
                        "part": section.part,
                        "item": section.item,
                        "section_title": section.title,
                        "chunk_type": "table",
                        "table_title": block.table_title,
                        "row_count": len(block.table_data),
                        "column_count": len(block.table_data[0]) if block.table_data else 0,
                    }

                    chunks.append(
                        ChunkItem(
                            part=section.part,
                            item=section.item,
                            sub_section=block.table_title or section.title,
                            breadcrumb=breadcrumb,
                            chunk_type="table",
                            chunk_index=global_index,
                            content=block.content,
                            embedding_input=embed_text,
                            metadata=table_meta,
                        )
                    )
                    global_index += 1

                else:
                    # Narrative block
                    paragraph = block.content.strip()
                    if not paragraph:
                        continue

                    if current_buffer_len + len(paragraph) > self.target_chunk_chars:
                        flush_text_buffer()

                    current_text_buffer.append(paragraph)
                    current_buffer_len += len(paragraph)

            # Flush any remaining text in this section (hard boundary at section end)
            flush_text_buffer()

        logger.info(f"Generated {len(chunks)} chunks across {len(sections)} sections.")
        return chunks
