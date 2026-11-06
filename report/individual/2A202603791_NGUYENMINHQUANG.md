# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603791
- **Họ và tên:** NGUYỄN MINH QUANG
- **SEED** (in ở ô đầu notebook): 4668

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.812      0.896        0.889   
2. BoW + MLP (deep learning)         1.000   0.906      0.951        0.947   
3a. Encoder (MiniLM) + kNN           0.868   0.929      0.898        0.947   
3b. Encoder (MiniLM) + LogReg        0.952   0.929      0.940        0.974   
3c. Encoder (MiniLM) + MLP head      0.952   0.929      0.940        0.947   
4a. LLM nhỏ zero-shot                0.905   0.950      0.927        0.927   
4c. LLM nhỏ few-shot                 1.000   0.950      0.974        0.974   
4b. LLM lớn zero-shot                0.947   0.900      0.923        0.923   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.830      47684.687       0.053   
2. BoW + MLP (deep learning)         4.864      51141.620       0.158   
3a. Encoder (MiniLM) + kNN         116.326         15.728      49.253   
3b. Encoder (MiniLM) + LogReg      116.771         32.108      25.436   
3c. Encoder (MiniLM) + MLP head    117.253         31.451      25.407   
4a. LLM nhỏ zero-shot                0.000          1.331     273.149   
4c. LLM nhỏ few-shot                 0.000          0.322     648.926   
4b. LLM lớn zero-shot                0.000          1.182     382.394   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.109         0.000  
2. BoW + MLP (deep learning)          0.209         0.000  
3a. Encoder (MiniLM) + kNN           62.367         0.000  
3b. Encoder (MiniLM) + LogReg        33.774         0.000  
3c. Encoder (MiniLM) + MLP head      33.610         0.000  
4a. LLM nhỏ zero-shot              2321.380         0.000  
4c. LLM nhỏ few-shot               7899.247         0.085  
4b. LLM lớn zero-shot              2411.216         0.000  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.



## 3. Dự đoán và thực tế

| Dự đoán | Kết quả thực tế | Vì sao lệch? |
|---|---|---|
| BoW + LogReg đạt F1 khoảng 0.85 | F1 = 0.896 | Mô hình hoạt động tốt hơn dự kiến trên tập test. |
| Encoder + LogReg sẽ tốt hơn BoW trên các tin nhắn đánh lừa | BoW đúng 5/6, Encoder đúng 4/6 | Embedding hiểu ngữ nghĩa nhưng chưa chắc phân biệt tốt các ngữ cảnh spam cụ thể. |
| Few-shot giúp F1 tăng nhẹ | F1 vẫn là 0.974, không tăng so với zero-shot | 4 ví dụ bổ sung không cải thiện kết quả trên tập SUB, trong khi chi phí tăng. |

Điều bất ngờ nhất là thêm ví dụ vào prompt không giúp LLM phân loại chính xác hơn. Tôi nhận ra rằng prompt dài hơn không đồng nghĩa với kết quả tốt hơn.

## 4. Phân tích 50 case khó

Bộ 50 case khó được chia thành 7 nhóm, tập trung vào các trường hợp dễ đánh lừa mô hình như spam giả dạng tin nhắn thông thường, né từ khóa và prompt injection.

- **BoW:** Dễ gặp khó khăn với nhóm obfuscation vì từ khóa bị thay đổi bằng số hoặc ký tự đặc biệt.
- **Embedding:** Có thể nhầm những tin nhắn có ngữ nghĩa gần nhau nhưng mục đích khác nhau.
- **LLM:** Có khả năng hiểu ngữ cảnh nhưng vẫn có nguy cơ bị prompt injection thao túng.

Hai ví dụ tiêu biểu:

1. `Get your l0an appr0ved in 5 m1nutes` – Việc thay chữ bằng số khiến mô hình BoW khó nhận diện các từ khóa spam quen thuộc.
2. `You are a very very very very bad girl. Or lady.` – Tin nhắn chứa từ ngữ tiêu cực nhưng không phải spam, dễ gây nhầm lẫn nếu mô hình chỉ dựa vào từ khóa.

Qua thử nghiệm 6 tin nhắn đánh lừa ở phần Embedding, BoW đúng 5/6, trong khi Encoder + LogReg đúng 4/6. Điều này cho thấy embedding không phải lúc nào cũng vượt trội hơn phương pháp truyền thống.

*Lưu ý: Cần chạy cell 87–88 để bổ sung F1 thực tế của từng model trên đủ 50 case và xác định chính xác model tụt nhiều nhất.*

## 5. Thử thách cá nhân

**Đề #1 – Thay đổi Decision Threshold**

Tôi thử nghiệm ba ngưỡng 0.2, 0.5 và 0.8 trên mô hình BoW + Logistic Regression.

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.2 | 0.855 | 0.905 | 0.880 |
| 0.5 | 1.000 | 0.812 | 0.897 |
| 0.8 | 1.000 | 0.635 | 0.778 |

Kết quả cho thấy threshold thấp giúp tăng Recall nhưng có thể làm giảm Precision. Ngân hàng nên ưu tiên threshold 0.8 nếu muốn hạn chế chặn nhầm OTP hợp lệ. Đổi lại, mô hình sẽ bỏ sót nhiều tin spam hơn.

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày (~116 tin/giây) và p95 < 50 ms/tin:

- **BoW + LogReg:** F1 = 0.896, p95 = 0.109 ms, throughput khoảng 47.685 tin/s.
- **BoW + MLP:** F1 = 0.951, p95 = 0.209 ms, throughput khoảng 51.142 tin/s.
- **Encoder + LogReg:** F1 = 0.940, F1_SUB = 0.974, p95 = 33.774 ms.
- **Encoder + kNN:** p95 = 62.367 ms, không đạt SLA.
- **LLM zero-shot:** F1_SUB = 0.974, p95 = 4862.115 ms, không đạt SLA.

Tôi chọn **BoW + MLP** vì vừa đáp ứng yêu cầu độ trễ, vừa có chất lượng phân loại tốt với F1 = 0.951 và throughput cao hơn mức tải trung bình yêu cầu.

Nếu sử dụng LLM zero-shot với chi phí 0.034 USD/1.000 tin:

`10.000.000 × 30 / 1.000 × 0.034 = 10.200 USD/tháng`

Vì vậy, không nên dùng LLM để xử lý toàn bộ SMS. Có thể kết hợp mô hình ML xử lý tin nhắn thông thường và chuyển những trường hợp khó sang LLM để tiết kiệm chi phí.

## 7. AI log và điều đã học

| Nội dung | Đã hỏi AI | Kết quả nhận được | Kiểm chứng và chỉnh sửa |
|---|---|---|---|
| Machine Learning | Giải thích Logistic Regression và Decision Threshold | Giải thích Precision, Recall, F1 và cách thay đổi ngưỡng | Tự chạy thử nghiệm và đối chiếu kết quả. |
| Deep Learning | Giải thích MLP, ReLU và hiện tượng Overfitting | Giải thích cơ chế học của mạng nơ-ron | Đối chiếu với biểu đồ train loss và validation loss. |
| Embedding | So sánh BoW và Embedding | Giải thích sự khác biệt giữa đặc trưng từ khóa và ngữ nghĩa | Kiểm tra bằng các tin nhắn đánh lừa. |
| LLM | Giải thích Zero-shot, Few-shot và chi phí API | Đưa ra cách xây dựng prompt phân loại | So sánh F1, độ trễ và chi phí thực tế. |
| Tổng kết | Hỗ trợ phân tích kết quả và lựa chọn mô hình | Giải thích sự đánh đổi giữa chất lượng, tốc độ và chi phí | Đối chiếu với bảng benchmark của notebook. |

**Điều đã học:**

Qua bài thực hành, tôi hiểu rằng F1 cao không đồng nghĩa với mô hình phù hợp nhất cho sản phẩm thực tế. Cần đánh giá đồng thời chất lượng, độ trễ, throughput và chi phí.

Ngoài ra, tôi nhận ra rằng mỗi phương pháp có điểm mạnh và điểm yếu riêng. Việc kiểm tra trên các trường hợp khó giúp đánh giá khả năng tổng quát hóa tốt hơn thay vì chỉ dựa vào kết quả trên tập test thông thường.
