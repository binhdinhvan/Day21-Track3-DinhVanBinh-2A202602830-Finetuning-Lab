# Lab 21 — Evaluation Report

**Họ tên**: Đinh Văn Bình  **MSSV**: 2A202602830  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Colab Free T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây khớp chính xác với các file trong `results/`.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Lý do chọn model & dataset**:
- Base model `unsloth/Qwen3.5-4B` có khả năng hiểu tiếng Việt rất tốt, vừa vặn bộ nhớ GPU T4 16GB ở định dạng 16-bit (không cần lượng tử hóa 4-bit) và tuân thủ chặt chẽ cấu trúc prompt.
- Dataset 250 ticket CSKH tiếng Việt có nhãn khách quan rõ ràng cho 4 trường dữ liệu, cho phép đo đạc chính xác bằng hàm đánh giá định lượng mà không cần đến LLM judge, đảm bảo tính công bằng và có thể tái lập 100%.
- Giá trị `max_length=256` được chọn dựa trên số đo thực tế p95 = 98 tokens (làm tròn lên lũy thừa 2 an toàn là 256), giúp bao quát trọn vẹn 100% mẫu huấn luyện mà không lãng phí chi phí bộ nhớ đệm padding.

**Template có giữ khối `<think>` không?** Có — theo kết quả từ `results/template_check.json`, chat template của model giữ nguyên vẹn cặp thẻ `<think>` và nội dung suy luận (`open_tag_present: true`, `body_present: true`, verdict: `reasoning preserved — safe to train on traces`). Do nhãn huấn luyện của dataset là JSON thuần túy, template đóng khối `<think>` rỗng ngay trong prompt dẫn đầu, không làm rò rỉ suy luận vào phần nhãn được tính loss.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Đoạn trích các dòng đầu của phần được tính loss:

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt câu hỏi (`<|im_start|>user...`) đã được gán mask token ID -100 hoàn toàn, bảo đảm loss chỉ tối ưu trên nội dung phản hồi của assistant.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3300.2 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1022.5 |
| (c) LoRA fine-tune | 0.970 | 0.656 | 1.000 | 1431.9 |

**(b) có thật sự mạnh hơn (a) không?** Có. Điểm target của prompt (b) đạt 0.765 (vượt trội hoàn toàn so với 0.000 của prompt ngây thơ a), tỷ lệ định dạng JSON đúng đạt 1.000 (100%), và độ trễ giảm hơn 3 lần từ 3300.2 ms xuống 1022.5 ms do prompt (b) chỉ dẫn mô hình trả về trực tiếp JSON ngắn gọn mà không sinh các lời chào hỏi dẫn dắt lan man.

Tôi **không sửa** `OPTIMIZED_PROMPT`, giữ nguyên vẹn mã băm SHA `719e74d3b6232053` như ban đầu để bảo toàn tính trung thực và liêm chính của phép so sánh đối đầu trước khi huấn luyện mô hình.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6273 | 0.970 | 405.5 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5372 | 0.970 | 278.1 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5702 | 0.000 | 420.1 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 485.9 | 3.86 |

### Phân tích chi tiết:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về rank so với vị trí gắn adapter?**
Run `attn_only` đã được khớp chính xác ngân sách tham số với `correct` bằng cách tăng rank lên r=283 (32,456,704 so với 32,464,896 tham số, độ lệch chỉ 0.025%). Trên tập đánh giá target ở NB5 §4, `attn_only` đạt 0.970, hoàn toàn hoà với `correct` (0.970). Tuy nhiên, nếu xét theo train loss ở NB4 thì `attn_only` lại có loss thấp hơn đáng kể (0.5372 so với 0.6273 của `correct`). Thứ tự xếp hạng theo train loss hoàn toàn trái ngược với bản chất năng lực thực tế trên tác vụ, phản ánh hiện tượng dồn rank cực lớn vào một cụm module hẹp khiến mô hình overfit/ghi nhớ dữ liệu huấn luyện tốt hơn nhưng không cải thiện thêm năng lực tổng quát. Điều này chứng minh rằng rank cao không phải là đòn bẩy chất lượng chính; việc phân bổ adapter trên toàn bộ các lớp tuyến tính (`all-linear`) với rank vừa phải (r=16) mang lại biểu diễn cân bằng và an toàn hơn cho mô hình.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` hạ learning rate xuống 1e-5 (thang đo thông thường của full fine-tuning) thay vì 1e-4 của LoRA. Đường loss của `wrong_lr` gần như đi ngang suốt 30 steps và kết thúc ở mức 1.5702, khiến điểm target rơi thẳng về 0.000 và format đúng bằng 0.000. Nếu một kỹ sư chỉ nhìn vào loss cao và kết quả thất bại mà không nắm rõ learning rate, họ sẽ dễ dàng kết luận sai lầm rằng phương pháp LoRA không hiệu quả, kiến trúc mô hình không phù hợp hoặc tập dữ liệu 250 mẫu là quá ít để học được định dạng JSON. Thực chất, do các ma trận LoRA A và B được khởi tạo gần bằng 0 (hoặc Gaussian nhỏ), chúng đòi hỏi một tốc độ học lớn hơn gấp 10 lần so với full fine-tuning để các bước cập nhật gradient có đủ độ lớn dịch chuyển biểu diễn tiềm ẩn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` giảm mức chiếm dụng bộ nhớ đỉnh từ 8.78 GB xuống chỉ còn 3.86 GB, tức tiết kiệm được tới 4.92 GB VRAM (~56% dung lượng bộ nhớ). Tuy nhiên, mức tiết kiệm này phải trả giá bằng hai yếu tố: thời gian huấn luyện kéo dài thêm 20% (485.9s so với 405.5s do overhead giải lượng tử hóa on-the-fly) và điểm số target bị giảm nhẹ từ 0.970 xuống 0.940. Với việc GPU T4 trên Colab cung cấp tới 14.6 GB VRAM khả dụng, cấu hình 16-bit LoRA nguyên bản (chỉ chiếm 8.78 GB) hoàn toàn vừa vặn thoải mái mà không hề đối mặt với nguy cơ OOM. Do đó, các số đo thực nghiệm ủng hộ hoàn toàn khuyến nghị của nhà sản xuất: không nên sử dụng QLoRA trên họ mô hình này khi tài nguyên phần cứng vẫn còn dư dả, nhằm tránh sai số lượng tử hóa và bảo toàn tối đa độ chính xác của tác vụ.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.136` · `valid_trace_rate = 0.0`

### Diễn giải phán quyết:
Cổng hồi quy đưa ra phán quyết `FAILED` bởi vì năng lực ngôn ngữ tổng quát của mô hình đã bị suy giảm 0.136 (điểm `regression` tụt từ 0.7911 của base model xuống còn 0.6556 ở bản fine-tune), vượt xa ngưỡng dung sai tối đa cho phép là 0.020. Mặc dù trên tác vụ mục tiêu, mô hình fine-tune đạt bước nhảy vọt rất ấn tượng (target tăng từ 0.765 lên 0.970, tức `target Δ = +0.205`), nó đã mắc phải hiện tượng quên lãng thảm họa (catastrophic forgetting). Toàn bộ 225 mẫu huấn luyện chỉ tập trung duy nhất vào cấu trúc JSON của ticket CSKH mà không hề có bất kỳ dữ liệu giữ chân (replay buffer) nào thuộc miền tri thức phổ thông. Điều này khiến các trọng số adapter LoRA điều chỉnh biểu diễn tiềm ẩn theo hướng chuyên biệt hóa quá mức, làm sai lệch khả năng thực thi các chỉ dẫn tổng quát ngoài miền huấn luyện. Đây là một kết quả thực nghiệm hoàn toàn bình thường và mang tính giáo khoa sâu sắc: nó chứng minh rằng việc đánh giá toàn diện qua 4 nhóm chỉ số (đặc biệt là nhóm regression) là tối quan trọng để ngăn chặn việc đưa vào vận hành một mô hình tưởng chừng như rất tốt nhưng thực chất đã bị tổn thương năng lực nền tảng.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Shop ơi, mình đặt nồi chiên không dầu mã đơn OD169066. Trả hàng. Mong shop phản hồi. Mình vẫn tin tưởng shop. | `doi_tra`, `trung_binh`, `nồi chiên không dầu`, `tich_cuc` | `sentiment: trung_tinh` | `sentiment: tich_cuc` | ✅ **FT thắng**: FT phân tích chính xác sắc thái "tin tưởng shop" là tích cực, trong khi prompt (b) bị từ "Trả hàng" làm thiên lệch sang trung tính. |
| 2 | Shop ơi, mình đặt sạc dự phòng mã đơn VN727311. Còn hàng không. Tôi cần trước ngày mai. Shop xem giúp. | `hoi_thong_tin`, `cao`, `sạc dự phòng`, `trung_tinh` | Sinh thừa thẻ markdown ` ```json ` | Trả về JSON thuần 100% | ✅ **FT thắng**: FT tuân thủ tuyệt đối định dạng JSON sạch không có text dẫn dụ thừa thãi, trong khi prompt (b) thỉnh thoảng sinh kèm markdown wrapper. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `urgency: thap` | `urgency: trung_binh` | ❌ **FT thua**: FT bị bias gán urgency thành `trung_binh`, bỏ qua cụm từ "Khi nào tiện" biểu thị mức độ khẩn cấp thấp. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `urgency: thap` | `urgency: trung_binh` | ❌ **FT thua**: Tương tự ca 3, mô hình fine-tune không nhạy bén với chỉ dấu giảm độ khẩn cấp bằng base model kèm prompt tối ưu. |
| 5 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. Mong shop phản hồi. Nhờ shop kiểm tra. | `hoi_thong_tin`, `trung_binh`, `ốp lưng điện thoại`, `trung_tinh` | `intent: san_pham_loi` | `intent: hoi_thong_tin` | ✅ **FT thắng**: FT phân loại chính xác intent hỏi giá, trong khi prompt (b) bị từ khóa phụ "kiểm tra" đánh lừa thành lỗi sản phẩm. |

**Mẫu chung ở các ca FT thua**:
Trong toàn bộ các mẫu mà mô hình fine-tune bị trừ điểm (đạt điểm 0.75 thay vì 1.0), lỗi xuất hiện đồng nhất ở trường `urgency` khi ticket xuất hiện cụm từ mang tính nhã nhặn "Khi nào tiện.". Do trong tập dữ liệu huấn luyện, nhãn `urgency: trung_binh` chiếm tỷ trọng áp đảo và thường đi kèm với các câu trao đổi lịch sự, mô hình LoRA đã hình thành một thiên kiến cố định gán mặc định về mức trung bình. Ngược lại, Base model khi kết hợp cùng Prompt (b) có định nghĩa quy tắc rõ ràng lại giữ được khả năng suy luận ngữ nghĩa linh hoạt hơn đối với các sắc thái từ vựng tinh tế này.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Từ các bằng chứng thực nghiệm thu thập được trong Lab 21, tôi kết luận rằng **chưa nên đưa bản LoRA fine-tune này vào môi trường sản xuất dưới dạng một mô hình đa năng độc lập**, mà chỉ nên sử dụng trong một pipeline microservice chuyên biệt khép kín cho tác vụ triage CSKH, hoặc cần phải huấn luyện lại với replay data. 

Về mặt tích cực, bản fine-tune đã chứng minh ưu thế vượt bậc trên tác vụ mục tiêu: độ chính xác trích xuất trường đạt 97.0% (vượt xa baseline prompt tối ưu 76.5%), tỷ lệ chuẩn hóa format JSON đạt 100%, và loại bỏ hoàn toàn nhu cầu truyền tải các prompt hướng dẫn dài dòng giúp tiết kiệm đáng kể chi phí token đầu vào. Tuy nhiên, việc mô hình bị trượt cổng hồi quy (`regression Δ = -0.136`) do hiện tượng quên lãng thảm họa là một rủi ro hiện hữu nếu mô hình phải tiếp nhận các yêu cầu tương tác ngoài luồng.

Đòn bẩy thực sự quyết định sự thành bại trong lab này không phải là rank LoRA cực cao hay kỹ thuật nén lượng tử, mà chính là **tính đúng đắn của loss mask** (đảm bảo chỉ học trên câu trả lời), **thang đo learning rate thích hợp cho LoRA (1e-4)**, và **chiến lược gắn adapter toàn diện (`text-linear`)**. Nếu muốn triển khai bản mô hình này một cách an toàn và bền vững nhất, giải pháp bắt buộc là trộn thêm từ 2% đến 5% dữ liệu chỉ dẫn đa năng vào tập huấn luyện để duy trì năng lực suy luận nền tảng trong khi vẫn giữ trọn vẹn độ chính xác trích xuất 97%.

### Ba điều tôi học được:
1. **Loss mask là nền tảng cốt lõi của SFT**: Nếu không kiểm tra mask proof bằng cách giải mã ngược token ID -100, việc vô tình tính loss trên cả prompt câu hỏi (`everything`) sẽ khiến mô hình học cách lặp lại câu hỏi của người dùng và làm hỏng toàn bộ pipeline mà không hề có thông báo lỗi cảnh báo nào.
2. **Không bao giờ dùng train loss làm thước đo phán quyết năng lực**: Run `attn_only` (r=283) đạt loss huấn luyện thấp hơn `correct` (0.5372 so với 0.6273) nhưng khi đo đạc trên tác vụ kiểm thử thực tế thì kết quả chỉ là hoà (0.970). Loss thấp trên tập dữ liệu nhỏ thường chỉ phản ánh sự ghi nhớ cục bộ chứ không đại diện cho năng lực tổng quát hóa.
3. **Prompt engineering là mốc chuẩn bắt buộc phải đánh bại**: Một prompt được thiết kế bài bản với schema chi tiết và ví dụ minh họa có thể mang lại độ chính xác tới 76.5% với chi phí triển khai bằng 0. Huấn luyện mô hình chỉ thực sự có giá trị khi nó chứng minh được khả năng vượt qua mốc prompt tối ưu này một cách rõ ràng và kiểm soát được rủi ro quên lãng tri thức nền tảng.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. Trộn thêm 3% dữ liệu replay đa năng (lấy từ tập OpenHermes hoặc Alpaca tiếng Việt) vào tập dữ liệu huấn luyện để chạy lại NB3 và NB5, nhằm khắc phục hiện tượng catastrophic forgetting và đưa cổng hồi quy về trạng thái `PASSED`.
2. Thực hiện bài thực hành B1 trong NB6 (`06_merge_and_serve.py`): Thực hiện merge trọng số adapter trực tiếp vào base model, đo đạc độ trễ suy luận thực tế khi overhead bằng 0 và thử nghiệm cơ chế hot-swap adapter giữa nhiều tác vụ chăm sóc khách hàng khác nhau.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
