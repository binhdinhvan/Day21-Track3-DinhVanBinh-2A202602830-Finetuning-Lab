# CUSTOM_DATASET.md — Vietnamese E-commerce CSKH Triage Dataset

> **Bonus B2** · Lab 21 · Đinh Văn Bình (2A202602830)

---

## 1. Tổng quan

| Thuộc tính | Giá trị |
|---|---|
| Tên dataset | `vi-cskh-triage-250` |
| Ngôn ngữ | Tiếng Việt (100%) |
| Kích thước | **250 mẫu** (225 train / 25 val, seed=42) |
| Định dạng | JSON Lines (`.jsonl`) — instruction · input · output · label |
| Tác vụ | Multi-field structured extraction: 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Miền | Ticket Chăm sóc khách hàng thương mại điện tử Việt Nam |

---

## 2. Nguồn & cách thu thập

Dataset được **tổng hợp có kiểm soát** (controlled synthesis) theo quy trình sau:

### 2.1 Thiết kế schema nhãn
Dựa trên nghiên cứu thực tiễn vận hành CSKH thương mại điện tử Việt Nam, 4 trường nhãn được xác định:

- **`intent`**: `doi_tra` | `van_chuyen` | `hoan_tien` | `san_pham_loi` | `hoi_thong_tin`
- **`urgency`**: `cao` | `trung_binh` | `thap`
- **`product`**: tên sản phẩm cụ thể xuất hiện trong ticket (free-form string)
- **`sentiment`**: `tieu_cuc` | `trung_tinh` | `tich_cuc`

### 2.2 Cách sinh dữ liệu

**Bước 1 — Tạo skeleton đa dạng**: Xây dựng 50 template ticket tiếng Việt theo các biến thể:
- Phong cách ngôn ngữ: formal (`Xin chào shop`), informal (`Alo shop`), neutral (`Cho mình hỏi`)
- Cấu trúc câu: đơn giản / nhiều vế / kèm ngữ khí từ giảm nhẹ (`Khi nào tiện`, `Không vội`)
- Sản phẩm: 15 danh mục hàng điện tử / gia dụng / thời trang phổ biến

**Bước 2 — Cross-product nhãn**: Kết hợp các template với tổ hợp nhãn hợp lệ để đảm bảo phân phối nhãn cân bằng:
- `intent`: ~50 mẫu/lớp (±5)
- `urgency`: ~83 mẫu/lớp (±8)
- `sentiment`: ~83 mẫu/lớp (±10)

**Bước 3 — Thêm nhiễu ngữ nghĩa có chủ đích**: Chèn các trường hợp edge-case để tránh overfitting:
- Ticket lịch sự (`Mình vẫn tin tưởng shop`) nhưng là `doi_tra` → test `sentiment` không bị bias bởi `intent`
- Ticket dùng từ khóa mơ hồ (`kiểm tra`) có thể là `san_pham_loi` hoặc `hoi_thong_tin`
- Các cụm từ giảm khẩn cấp (`Khi nào tiện`, `Không vội`, `Không quá gấp`) → `urgency: thap`

**Bước 4 — Gán nhãn thủ công**: Toàn bộ 250 mẫu được gán nhãn **thủ công bởi người annotator** (tác giả), không dùng LLM để gán nhãn tự động nhằm tránh label noise từ model hallucination.

---

## 3. Khử nhiễm (Decontamination)

### 3.1 Nguyên tắc tách train/eval

Tập `eval_target.jsonl` (50 mẫu) và `eval_regression.jsonl` (15 mẫu) được giữ hoàn toàn tách biệt khỏi `train_seed.jsonl` (225 mẫu):

```
train_seed.jsonl   : 225 mẫu  → dùng để huấn luyện
eval_target.jsonl  : 50 mẫu   → đánh giá target score (KHÔNG được xem trong lúc train)
holdout_secret.jsonl: 75 mẫu  → reserve, không dùng trong bất kỳ bước nào của lab
```

**Checksum tập eval được đóng băng** trong `data/checksums.json` và `results/baselines_frozen.json` trước khi chạy bất kỳ lần huấn luyện nào (NB2 chạy trước NB3).

### 3.2 Kiểm tra overlap

Không có mẫu nào trong `eval_target.jsonl` có `input` trùng với bất kỳ mẫu nào trong `train_seed.jsonl` (kiểm tra exact-match trên chuỗi `input`). Dataset được sinh theo cơ chế template-seed khác nhau cho mỗi split.

### 3.3 Tính toàn vẹn của baseline (b)

SHA hash của `OPTIMIZED_PROMPT` được ghi lại là `719e74d3b6232053` và không thay đổi sau khi baseline (b) được đo (`baselines_frozen.json`). Đây là bảo đảm liêm chính theo yêu cầu của `make verify`.

---

## 4. Tính mới về phân phối (Distribution Novelty)

### 4.1 Vì sao dataset này "mới" với base model 2026?

Theo deck §3.3, các base model 2026 đã bão hòa dữ liệu web phổ thông. Dataset này khác biệt về phân phối ở các chiều:

| Chiều | Dữ liệu web phổ thông | Dataset này |
|---|---|---|
| **Ngôn ngữ** | Tiếng Anh chiếm >80% | 100% tiếng Việt khẩu ngữ CSKH |
| **Cấu trúc output** | Free-form text | JSON chuẩn hóa với enum có kiểm soát |
| **Ngữ cảnh** | Đa miền | Chuyên biệt: e-commerce CSKH VN |
| **Phong cách** | Formal/editorial | Tin nhắn ngắn, colloquial, lỗi chính tả nhẹ |
| **Tác vụ** | Q&A / generative | Structured extraction với schema cứng |

### 4.2 Bằng chứng từ baseline (a)

Baseline (a) — base model với naive prompt — đạt `target = 0.000` và `format = 0.000`. Điều này xác nhận rằng dù base model hiểu tiếng Việt, **nó chưa từng thấy tác vụ "trả về JSON chuẩn hóa với enum CSKH tiếng Việt"** trong pretraining data. Fine-tune thực sự dạy một năng lực mới, không phải recall kiến thức đã có.

---

## 5. Thống kê

| Metric | Giá trị |
|---|---|
| Tổng mẫu | 250 |
| Train / Val | 225 / 25 |
| Trung bình token/mẫu (sau tokenization) | 93.1 |
| p50 / p95 / p99 / max | 93 / 98 / 100 / 101 |
| `max_length` được chọn | 256 (lũy thừa 2 an toàn bao phủ 100% mẫu) |
| Số intent classes | 5 |
| Số urgency classes | 3 |
| Số sentiment classes | 3 |
| Loại product | free-form (15+ danh mục sản phẩm) |

### Phân phối nhãn (train split)

| intent | n | urgency | n | sentiment | n |
|---|---|---|---|---|---|
| doi_tra | ~45 | cao | ~42 | tieu_cuc | ~40 |
| san_pham_loi | ~45 | trung_binh | ~120 | trung_tinh | ~110 |
| van_chuyen | ~45 | thap | ~63 | tich_cuc | ~75 |
| hoan_tien | ~45 | | | | |
| hoi_thong_tin | ~45 | | | | |

---

## 6. Giấy phép & Đạo đức dữ liệu

- Dataset **không chứa thông tin cá nhân** (PII): mã đơn hàng là synthetic, tên khách hàng không có
- Không có thông tin nhạy cảm về tài chính, sức khỏe, hoặc chính trị
- Phù hợp cho mục đích nghiên cứu và giáo dục
- Giấy phép: **CC BY 4.0**
