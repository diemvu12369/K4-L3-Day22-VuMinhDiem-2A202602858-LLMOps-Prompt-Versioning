# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Vũ Minh Điềm — 2A202602858
**LangSmith project:** `day22-lab-vuminhdiem`
**Prompt Hub:** `vu-minh-diem-rag-v1`, `vu-minh-diem-rag-v2`
**Provider:** Gemini free tier. Embeddings: `gemini-embedding-001`. LLM: xem mục "Môi trường chạy".

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | ≥ 50 trace `rag-query` (mỗi trace có retriever → prompt → LLM → parser) |
| `02_prompt_hub.png` | 2 prompt trên Prompt Hub |
| `02_ab_routing_log.txt` | Log push/pull Hub + routing: V1 = 19 câu, V2 = 31 câu (MD5 của `request_id`) |
| `03_ragas_scores.png` | Bảng so sánh V1 và V2 trên terminal |
| `03_ragas_report.json` | Báo cáo RAGAS (bản sao `data/ragas_report.json`) |
| `03_ragas_log.txt` | Log phần chấm RAGAS và bảng so sánh |
| `04_pii_demo_log.txt` | 7 test case PII (email, phone, SSN, thẻ, nhiều loại PII, câu trả lời LLM, văn bản sạch) |
| `04_json_demo_log.txt` | 6 test case JSON (hợp lệ, fences, nháy đơn, dấu phẩy thừa, lỗi kết hợp, không sửa được → fallback) |

---

## Kết quả RAGAS

| Metric | V1 (ngắn gọn) | V2 (chuyên gia, có cấu trúc) | Winner |
|---|---:|---:|---|
| faithfulness | **0.9884** (n=43) | 0.9786 (n=33) | V1 |
| answer_relevancy | **0.8502** (n=50) | 0.8493 (n=50) | V1 (chênh không đáng kể) |
| context_recall | 1.0000 (n=50) | 1.0000 (n=50) | Hoà |
| context_precision | 0.9650 (n=50) | 0.9650 (n=50) | Hoà |

Faithfulness ≥ 0.9 ở **cả 2** phiên bản, vượt mục tiêu ≥ 0.8.

## Phân tích: vì sao V1 cao hơn V2

**1. `context_recall` và `context_precision` giống hệt nhau. Đây là kết quả đúng, không phải lỗi.**
Hai chỉ số này chỉ đo bước retrieval: câu hỏi → FAISS top-3 chunk, so với đáp án chuẩn. Prompt không ảnh hưởng đến bước này. V1 và V2 dùng chung vectorstore, chung `k=3`, chung câu hỏi, nên nhận về đúng cùng 3 đoạn context. Prompt chỉ tác động đến **câu trả lời**, tức là đến `faithfulness` và `answer_relevancy`.

**2. Faithfulness: V1 thắng vì nói ít hơn, nên có ít câu khẳng định để sai hơn.**
V2 trả lời dài gần gấp đôi V1:

| | Số từ trung bình | Số câu trung bình |
|---|---:|---:|
| V1 | 37 | 2.1 |
| V2 | 72 | 3.2 |

RAGAS tính faithfulness bằng cách tách câu trả lời thành các câu khẳng định (claim), rồi đếm tỉ lệ claim được context chứng minh. Prompt V2 yêu cầu 3–5 câu và "supporting details". Khi context không đủ chi tiết cho chừng đó câu, model có xu hướng **lấp chỗ trống bằng kiến thức nền** hoặc câu diễn giải chung chung. Mỗi câu như vậy là một claim không có căn cứ.

Trong số sample đã chấm, 5 sample V2 có faithfulness < 1 (0.82–0.90), trong khi V1 chỉ có 2. Ví dụ với câu *"What is FAISS?"*, V2 viết thêm *"developed by Meta AI Research"* và *"applications like RAG pipelines and recommendation systems"*. Đây là những chi tiết đúng ngoài đời thực nhưng không có trong 3 chunk được retrieve, nên bị chấm là không được context hỗ trợ (0.82).

V1 bị ràng buộc "2–4 câu, reuse the wording of the context", nên gần như chép lại sự kiện từ context. Rủi ro bịa thêm thông tin vì vậy thấp hơn.

**3. Answer relevancy: hai phiên bản gần như hoà (0.850 so với 0.849).**
Cả hai đều trả lời đúng trọng tâm câu hỏi. Câu dài hơn của V2 không làm câu trả lời bám câu hỏi hơn: các câu phụ của V2 thường mở rộng sang ứng dụng hoặc ý nghĩa, nên "câu hỏi được sinh ngược" từ câu trả lời đôi khi lệch khỏi câu hỏi gốc. Lợi thế "đầy đủ hơn" của V2 vì vậy không chuyển thành điểm relevancy cao hơn.

**Kết luận:** với RAG cần độ trung thực cao, nên chọn **V1**: ngắn, bám context, rẻ hơn khoảng 50% token output. V2 chỉ đáng dùng khi người dùng cần giải thích dài. Khi đó nên sửa prompt V2 để "chỉ thêm chi tiết nếu context có", thay vì ép đủ 3–5 câu.

---

## Môi trường chạy và giới hạn (ghi chú trung thực)

Lab chạy hoàn toàn trên **Gemini free tier**, nên gặp các giới hạn sau:

- **Model sinh câu trả lời:** Bước 1–2 dùng `gemini-3.5-flash-lite`. Model này hết quota ngày (500 request) giữa chừng Bước 3, nên Bước 3 sinh lại toàn bộ câu trả lời V1/V2 bằng `gemini-3.1-flash-lite`. **Cả V1 và V2 ở Bước 3 dùng cùng một model**, nên phép so sánh vẫn công bằng.
- **Model chấm (judge) xoay vòng:** mỗi model chỉ có 20–500 request/ngày, nên `utils/llm_factory.py` tự chuyển sang model kế tiếp khi model hiện tại hết quota ngày. Kết quả: `gemini-3.1-flash-lite` (46 sample), `gemma-4-26b-a4b-it` (45), `gemini-3-flash-preview` (9). Sample V1 và V2 được xếp **xen kẽ** (v1#1, v2#1, v1#2, …), nên khi đổi model chấm, cả hai phiên bản bị ảnh hưởng gần như nhau (V1: 24/22/4, V2: 22/23/5).
- **`answer_relevancy` dùng `strictness=1`** (mặc định là 3): Gemini không hỗ trợ trả nhiều kết quả trong một request, nên mỗi mức strictness tốn thêm một request.
- **Faithfulness chưa đủ 50/50** (V1: 43, V2: 33): với model dự phòng Gemma, job faithfulness (prompt dài nhất) thường bị `504 DEADLINE_EXCEEDED` hoặc timeout. Ba chỉ số còn lại đủ 50/50 ở cả hai phiên bản. Điểm trung bình chỉ tính trên các sample chấm được (số lượng ghi ở `n_scored` trong report). Script hỗ trợ chạy tiếp: khi quota reset, chạy lại `python 03_ragas_evaluation.py` sẽ chỉ chấm các sample còn thiếu.
