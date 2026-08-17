import json
import logging
import os
import re
import sys
import traceback
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import hybrid_vec_knowgraph as rag


HOST = os.getenv("APP_HOST", "127.0.0.1")
PORT = int(os.getenv("APP_PORT", "8000"))
TOP_K = int(os.getenv("APP_TOP_K", "10"))
MAX_NEW_TOKENS = int(os.getenv("APP_MAX_NEW_TOKENS", "512"))
LLM_MODEL_ID = os.getenv("HF_MODEL_ID", rag.MODEL_ID)
NO_ANSWER_MESSAGE = "Không tìm thấy tài liệu phù hợp để trả lời câu hỏi."

logging.basicConfig(
    level=os.getenv("APP_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("qa_app")


@lru_cache(maxsize=1)
def get_llm():
    LOGGER.info("Loading LLM model: %s", LLM_MODEL_ID)
    tokenizer, model = rag.load_model(LLM_MODEL_ID)
    return tokenizer, model


def retrieve_hits(question: str):
    allowed_doc_ids = rag.graph_retrieve_documents(rag.driver, question)
    if isinstance(allowed_doc_ids, tuple):
        allowed_doc_ids = allowed_doc_ids[0]

    query_vector = rag.model.encode(question).tolist()
    if allowed_doc_ids:
        allowed_doc_ids = [str(doc_id).lower() for doc_id in allowed_doc_ids]
        hits = rag.vector_search_filtered(
            rag.qdrant,
            rag.COLLECTION_NAME,
            query_vector,
            allowed_doc_ids,
            limit=TOP_K,
        )
    else:
        hits = rag.qdrant.query_points(
            collection_name=rag.COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        )
    return rag.sort_hits_in_order(hits)


def make_final_answer_prompt(question: str, context: str) -> str:
    return f"""
Bạn là trợ lý hỏi đáp tài liệu hành chính tiếng Việt.

Nhiệm vụ: trả lời câu hỏi của người dùng bằng một câu trả lời cuối cùng, đúng và đầy đủ nhất.
  
Quy tắc bắt buộc:
- Chỉ sử dụng thông tin có trong CONTEXT.
- Không suy đoán, không thêm kiến thức ngoài tài liệu.
- Tổng hợp các bằng chứng liên quan thành một câu trả lời mạch lạc.
- Nếu câu hỏi hỏi văn bản/quyết định nào ban hành một quy chế, ưu tiên các trường so_quyet_dinh, ngay_ban_hanh và tom_tat_tai_lieu trong CONTEXT.
- Nếu đoạn OCR có số quyết định mâu thuẫn với so_quyet_dinh trong metadata, ưu tiên so_quyet_dinh vì OCR có thể nhiễu.
- Không liệt kê nhiều phương án trả lời.
- Không hiển thị mã nguồn tham chiếu như [E1], [E2], doc_id, chunk, file_url.
- Không viết lời dẫn như "Dựa trên context" hoặc "Câu trả lời là".
- Nếu CONTEXT không đủ dữ liệu để trả lời, chỉ trả lời: "{NO_ANSWER_MESSAGE}"

CONTEXT:
{context}

CÂU HỎI:
{question}

CÂU TRẢ LỜI CUỐI CÙNG:
"""


def clean_final_answer(answer: str) -> str:
    answer = (answer or "").strip()
    if not answer:
        return ""

    stop_markers = (
        "\nCONTEXT:",
        "\nQUESTION:",
        "\nCÂU HỎI:",
        "\nYÊU CẦU TRẢ LỜI:",
    )
    for marker in stop_markers:
        idx = answer.upper().find(marker.upper())
        if idx != -1:
            answer = answer[:idx].strip()

    answer = re.sub(r"(?im)^\s*(assistant|user|system)\s*:\s*", "", answer)
    answer = re.sub(r"(?im)^\s*câu trả lời(?: cuối cùng)?\s*:\s*", "", answer)
    answer = re.sub(r"\s*\[E\d+\]\s*", " ", answer)
    answer = re.sub(r"(?im)^\s*(doc_id|doc_no|chunk_index|file_url|nguồn)\s*:.*$", "", answer)

    lines = []
    seen = set()
    for raw_line in answer.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)

    answer = "\n".join(lines)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip(" :-\n\t")
    return answer


def collect_file_urls(hits) -> list[str]:
    file_urls = []
    seen = set()
    for hit in hits:
        payload = getattr(hit, "payload", {}) or {}
        file_url = (payload.get("file_url") or "").strip()
        if not file_url or file_url in seen:
            continue
        seen.add(file_url)
        file_urls.append(file_url)
    return file_urls


def answer_question(question: str) -> dict:
    question = question.strip()
    if not question:
        raise ValueError("Vui lòng nhập câu hỏi.")

    hits = retrieve_hits(question)

    if not hits:
        return {
            "answer": NO_ANSWER_MESSAGE,
            "file_urls": [],
        }

    file_urls = collect_file_urls(hits)
    context = rag.build_context_from_payloads(hits, top_k=TOP_K, query=question)
    prompt = make_final_answer_prompt(question, context)
    tokenizer, model = get_llm()
    decoded = rag.chat_generate(
        tokenizer,
        model,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    answer = clean_final_answer(rag.extract_answer(decoded, prompt))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "answer": answer or NO_ANSWER_MESSAGE,
        "file_urls": file_urls,
    }


HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Qdrant + Neo4j QA</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64748b;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --danger: #b42318;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .wrap {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    main {
      padding: 24px 0 40px;
    }
    .layout {
      max-width: 820px;
      margin: 0 auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .ask {
      padding: 18px;
    }
    label {
      display: block;
      margin-bottom: 10px;
      font-weight: 700;
    }
    textarea {
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      line-height: 1.5;
      outline: none;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.14);
    }
    .actions {
      margin-top: 12px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      border: 0;
      border-radius: 8px;
      min-height: 42px;
      padding: 0 16px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .status {
      color: var(--muted);
      font-size: 14px;
    }
    .answer {
      margin-top: 18px;
      padding: 18px;
    }
    .answer h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .answer-text {
      white-space: pre-wrap;
      line-height: 1.6;
    }
    .file-download {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }
    .file-download h3 {
      margin: 0 0 10px;
      font-size: 15px;
    }
    .file-links {
      display: grid;
      gap: 8px;
    }
    .file-link {
      display: block;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      line-height: 1.4;
    }
    .error {
      color: var(--danger);
      white-space: pre-wrap;
    }
    a { color: var(--accent-strong); overflow-wrap: anywhere; }
    @media (max-width: 900px) {
      .topbar { align-items: flex-start; flex-direction: column; padding: 14px 0; }
      .meta { white-space: normal; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>Website hỏi đáp tài liệu</h1>
      <div class="meta">Hybrid search: Qdrant + Neo4j + LLM</div>
    </div>
  </header>
  <main class="wrap">
    <div class="layout">
      <section>
        <form class="panel ask" id="qa-form">
          <label for="question">Câu hỏi</label>
          <textarea id="question" name="question" placeholder="Nhập câu hỏi cần tra cứu..." required></textarea>
          <div class="actions">
            <button id="submit" type="submit">Gửi câu hỏi</button>
            <span class="status" id="status"></span>
          </div>
        </form>
        <section class="panel answer" aria-live="polite">
          <h2>Câu trả lời</h2>
          <div id="answer" class="answer-text">Chưa có câu hỏi.</div>
          <div id="file-download" class="file-download" hidden>
            <h3>File_URL</h3>
            <div id="file-links" class="file-links"></div>
          </div>
        </section>
      </section>
    </div>
  </main>
  <script>
    const form = document.getElementById("qa-form");
    const question = document.getElementById("question");
    const submit = document.getElementById("submit");
    const statusEl = document.getElementById("status");
    const answerEl = document.getElementById("answer");
    const fileDownloadEl = document.getElementById("file-download");
    const fileLinksEl = document.getElementById("file-links");

    function renderFileUrls(fileUrls) {
      fileLinksEl.replaceChildren();
      if (!fileUrls || !fileUrls.length) {
        fileDownloadEl.hidden = true;
        return;
      }

      fileUrls.forEach((fileUrl, index) => {
        const link = document.createElement("a");
        link.className = "file-link";
        link.href = fileUrl;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.textContent = `Tải file ${index + 1}: ${fileUrl}`;
        fileLinksEl.appendChild(link);
      });
      fileDownloadEl.hidden = false;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;

      submit.disabled = true;
      statusEl.textContent = "Đang truy vấn...";
      answerEl.className = "answer-text";
      answerEl.textContent = "Đang tìm kiếm bằng chứng và sinh câu trả lời.";
      renderFileUrls([]);

      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Truy vấn thất bại.");

        answerEl.textContent = data.answer || "Không có câu trả lời.";
        renderFileUrls(data.file_urls);
        statusEl.textContent = "Hoàn tất.";
      } catch (error) {
        answerEl.className = "error";
        answerEl.textContent = error.message;
        renderFileUrls([]);
        statusEl.textContent = "Có lỗi.";
      } finally {
        submit.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class QARequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGQA/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self.send_json({"status": "ok"})
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
            result = answer_question(payload.get("question", ""))
            self.send_json(result)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            LOGGER.error("Request failed: %s\n%s", exc, traceback.format_exc())
            self.send_json(
                {"error": "He thong dang gap loi khi xu ly cau hoi."},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def send_json(self, payload: dict, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str, status=HTTPStatus.OK):
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main():
    server = ThreadingHTTPServer((HOST, PORT), QARequestHandler)
    LOGGER.info("QA web app is running at http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
