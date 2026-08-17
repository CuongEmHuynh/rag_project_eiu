# Giải thích function SAHC-v2

Tất cả function đều có type hint và docstring ngay tại source. Tài liệu này giải
thích luồng gọi và mục đích để review nhanh mà không cần đọc ngược toàn bộ code.

## `models.py`

- `OCRBlock.block_id`: tạo provenance ID dạng `page_001_block_0012` từ trang và
  index; không phụ thuộc text OCR.
- `DocumentTree.root`: tra root node từ `root_id`, tránh truyền hai object không
  đồng bộ.
- Các dataclass còn lại không biến đổi dữ liệu; chúng định nghĩa contract giữa
  parser, table processor, packer, builder, storage và retrieval.

## `normalize.py`

- `normalize_ocr_text(text)`: Unicode NFC, NBSP, whitespace và newline; không sửa
  dấu hỏi/lỗi chính tả/nội dung pháp lý.
- `normalize_block_type(value)`: map `PLAIN_TEXT`, `figurecaption`, `discard`... về
  vocabulary chuẩn.
- `is_indexable_block(block)`: lọc page number/operator note/figure/stamp noise;
  block `abandon` chứa cơ quan, số, quyết định, căn cứ hoặc Điều vẫn được giữ.
- `sort_blocks_in_reading_order(blocks)`: sort page → bbox top → bbox left → block
  index; bbox thiếu được đặt sau block có vị trí trên cùng trang.

## `ocr_parser.py`

- `load_ocr_json(path)`: đọc UTF-8, đổi lỗi file/JSON thành `OCRParseError` rõ nghĩa.
- `parse_ocr_data(data)`: parse schema chuẩn và alias field; giữ `content_raw`, tạo
  `content_normalized`, chuẩn hoá pixel bbox khi page dimensions có sẵn.
- `group_blocks_by_page(blocks)`: cấp page-local context cho table continuation.
- `_parse_page_number`: hiểu số, `page_001` và fallback array order.
- `_parse_block_index`: ưu tiên `block_index/order/reading_order/id`.
- `_extract_content`: đọc `content_raw/content/text/ocr`; không invent text.
- `_parse_bbox`: xác thực 4 số xyxy, scale về 0..1 khi có width/height.
- `_as_positive_float`: bảo vệ page dimensions lỗi/null.

## `structure_parser.py`

- `detect_article_boundary`: regex `Điều` anchored ở đầu paragraph nên không split
  substring “ở điều 1”.
- `detect_clause_boundary`: chỉ nhận `1.`/`2)` khi caller đang trong Điều.
- `detect_point_boundary`: chỉ nhận `a)`/`b.` khi caller đang trong Điều.
- `is_legal_basis`: nhận `Căn cứ/Căn cử/Căn cứ vào/Theo đề nghị` tolerant OCR.
- `is_recipient_boundary`: nhận `Nơi nhận` và đóng Điều hiện hành.
- `is_signature_boundary`: dùng chức vụ rõ ràng hoặc short figure caption.
- `build_document_tree`: state machine chính xây root/metadata/preamble/decision/
  article/clause/point/table/recipients/signature; mọi block sau Điều thuộc Điều đó
  tới boundary mới.
- `iter_nodes`: DFS deterministic, có filter node type.
- `find_ancestor`: tìm structural parent gần nhất theo accepted types.
- `map_source_blocks_to_nodes`: nối table OCR block với node/Điều tương ứng.
- `_paragraph_segments`: tách newline để áp dụng anchored boundary cho từng đoạn.
- `_active_structural_parent`: chọn Article trước Decision/Preamble/Metadata/Root.
- `_append_text`: nối signature lines, cập nhật page/provenance.
- `_stable_node_id`: UUID5 deterministic, kể cả document ID không phải UUID.
- `_propagate_page_ranges`: lan page/source từ descendants lên parent.

## `table_parser.py`

- `GenericKeyValueTableSerializer.serialize`: fallback `header: value`, bỏ ô trống.
- `CourseTransferTableSerializer.serialize`: xuất hai nhóm “Môn học đã học” và
  “Môn học được chuyển” với 8 labels riêng.
- `parse_html_table`: BeautifulSoup parse, expand span, nhận header/data, phát hiện
  data row bị đặt trong `thead`, flatten schema và classify table.
- `parse_document_tables`: parse từng table block và thêm Article parent, bbox,
  section path.
- `score_table_continuation`: cộng/trừ điểm theo trang kề, vị trí bottom/top, số
  cột, parent, boundary mới và schema similarity; trả cả reasons để debug.
- `is_table_continuation`: so score với threshold config.
- `merge_table_continuation`: kế thừa header trang trước, nối rows/source IDs và
  đánh dấu `cross_page`.
- `reconstruct_cross_page_tables`: quét physical tables theo trang và tạo logical
  tables.
- `choose_table_serializer`: Strategy selection course-transfer hoặc generic.
- `serialize_table_row`: API semantic row; có thể thêm document/section context.
- `_expand_html_rows`: occupancy-grid mở rộng rowspan/colspan.
- `_header_row_indexes`: lấy `thead` hoặc chuỗi row toàn `th` đầu bảng.
- `_flatten_headers`: nối group/leaf label, loại label span lặp.
- `_looks_like_data_row`: dùng course code + numeric/grade để nhận continuation
  data bị gắn nhãn header.
- `_detect_course_transfer_schema`: nhận schema 8 cột và biến thể có STT.
- `_boundaries_before_table`: chặn merge nếu Article/Decision/Chapter/Recipients
  mới xuất hiện trước table trang kế.
- `_header_similarity`: Jaccard token similarity của hai schemas.
- `_label_value_lines`: chỉ serialize giá trị tồn tại.
- `_positive_span`, `_pad_row`: defensive HTML helpers.
- `_stable_table_id`: UUID5 theo source provenance.

## `token_counter.py`

- `TokenCounter.__init__`: resolve tokenizer/max length, ưu tiên public attributes.
- `TokenCounter.count`: đếm full text có special tokens.
- `encode_content`/`decode_content`: dùng riêng cho token-window fallback.
- `_encode`: hỗ trợ tokenizer `encode()` hoặc callable, luôn `truncation=False`.
- `_resolve_tokenizer`: fallback qua SentenceTransformer first module.
- `_resolve_max_seq_length`: bỏ sentinel model length cực lớn của HuggingFace.
- `RegexTokenizer.encode/decode`: tokenizer xấp xỉ chỉ test/debug offline.
- `_flatten_token_ids`: chuyển tensor/batch/list về `list[int]`.

## `token_packer.py`

- `pack_atomic_units`: đếm candidate retrieval text đầy đủ; không trộn parent,
  unit type, section path hoặc table; row mặc định đứng riêng.
- `split_oversized_atomic_unit`: thử paragraph/sentence trước.
- `_split_by_token_window`: fallback cuối có overlap cấu hình và warning; giảm
  window tới khi prefix + body thật sự vừa.
- `_semantic_parts`: simple Vietnamese-friendly paragraph/sentence splitter.
- `_can_pack_together`: semantic compatibility gate.
- `_default_candidate_text`: builder khi không có contextual prefix.
- `_to_packed_unit`: gộp text/page/provenance và lưu atomic IDs/fallback metadata.

## `chunk_builder.py`

- `build_document_chunks`: production entry đọc JSON và lấy tokenizer từ model.
- `build_document_chunks_v2`: alias versioned cho feature flag.
- `build_chunks_from_ocr_document`: orchestrator pure/testable cho toàn pipeline.
- `create_atomic_units`: biến prose nodes và từng logical table row thành units.
- `_build_parent_chunks`: tạo Article/Preamble/... parents và table secondary parents.
- `_build_child_chunks`: materialize packed units, UUID5, retrieval text/token count.
- `_draft_chunk_from_units`: draft dùng để packer đếm contextual prefix trước.
- `_primary_parent`: nâng Khoản/Điểm/paragraph lên Article context rộng.
- `_unit_type_for_node`: map node type sang child vocabulary.
- `_link_parent_children`: ghi child IDs vào primary/table parent store.
- `_document_id`, `_meta_value`: null-safe document metadata resolution.
- `_stable_uuid`: deterministic IDs theo document namespace.
- `_unique_sources`: provenance dedup giữ thứ tự.
- `_source_order`, `_atomic_sort_key`: interleave prose/table rows theo OCR order.
- `_page_from_sources`: giữ đúng page của row trong merged table.
- `_log_document_stats`: log page/block/noise/articles/tables/parents/children/tokens.

## `retrieval_text.py`

- `safe_meta`: null-safe string; không sinh chữ `None`.
- `build_context_lines`: chỉ thêm Summary/No/Author/Date/section có giá trị.
- `build_retrieval_text`: prefix + `Nội dung` + normalized body.
- `chunk_to_payload`: payload đầy đủ raw/normalized/retrieval/provenance/version.

## `validators.py`

- `ChunkValidationError.__init__`: giữ toàn bộ lỗi để debug một lượt.
- `validate_chunks`: chặn duplicate ID, child thiếu parent, empty/near-empty text,
  count lệch, overflow, table row thiếu table ID/section path.

## `integration.py`

- `resolve_ocr_json_path`, `resolve_v1_txt_path`: một chỗ cấu hình OCR path và
  tolerant UUID case.
- `load_v2_chunks`: strict JSON production entry.
- `load_chunks_by_version`: feature flag v1/v2 và explicit labeled fallback.
- `build_v1_fallback_chunks`: bọc baseline thành token-safe parent/child records.
- `embedding_children`: lọc record được encode.
- `build_parent_store`: parent payload map riêng, không zero-vector.
- `build_qdrant_points`: validate → encode retrieval text → PointStruct.
- `upsert_document_v2`: chỉ upsert collection đã tồn tại.
- `create_v2_collection_explicit`: operator-only create, từ chối `rag_document`.
- `_stable_uuid`: stable IDs cho fallback/integration.

## `retrieval.py`

- `deduplicate_children`: giới hạn hit cùng parent, giữ ranking.
- `expand_context`: `none/parent/siblings/adaptive`; table row ưu tiên table schema
  parent, prose ưu tiên Article parent.
- `embedding_search_v2`: vector query + Qdrant child filter + dedup + expansion.
- `build_sibling_store`: group offline children theo parent.
- `_payload`, `_score`, `_append_store_record`: adapters cho dict/Chunk/Qdrant hit.

## `legacy.py`

- `clean_data_v1`: exact-style baseline cleaning.
- `chunk_legal_document_v1`: regex Điều + character threshold cũ để benchmark;
  không được dùng như core v2.

## `debug.py`

- `chunks_to_debug_dict`: chia parent/child và tính token statistics.
- `chunks_to_markdown`: render hierarchy review-friendly.
- `write_debug_outputs`: ghi JSON/Markdown UTF-8.
- `build_argument_parser`: định nghĩa CLI và offline flag explicit.
- `main`: tải meta/model hoặc offline tokenizer rồi chạy pipeline.
- `_load_meta`: đọc JSON metadata hoặc fallback filename cho debug.

