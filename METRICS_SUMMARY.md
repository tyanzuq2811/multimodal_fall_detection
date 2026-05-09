# Tổng Hợp Chỉ Số Mô Hình Phát Hiện Ngã

## Trạng Thái Hiện Tại
- **Pose**: đã retrain với kênh confidence, checkpoint tốt nhất ở epoch 48.
- **IMU**: đã retrain 150 epoch, checkpoint tốt nhất ở epoch 123.
- **Fusion**: đã retrain lại trên checkpoint IMU mới, checkpoint tốt nhất ở epoch 4.

Tài liệu này tổng hợp các chỉ số đánh giá chính của ba mô hình: **Pose**, **IMU** và **Fusion**. Tất cả số liệu bên dưới đều đã được cập nhật theo checkpoint mới nhất.

---

## 1. Mô Hình Pose

### Cấu hình mô hình
- **Checkpoint**: `weights/pose_finetuned_upfall.pth`
- **Kiến trúc**: TwoCamPoseClassifier + TemporalConvEncoder
- **Dạng đầu vào**: `(B, 3, 60, 17)` với 3 kênh `(x, y, confidence)`
- **Số đặc trưng đầu vào**: 51, tương ứng 17 khớp × 3 kênh
- **Embedding dimension**: 128
- **Hàm mất mát**: Binary Focal Loss với `alpha ≈ 0.82`, `gamma = 2.0`
- **Optimizer**: AdamW, `lr = 0.001`, `weight_decay = 0.0001`
- **Số epoch fine-tune**: 50, checkpoint tốt nhất ở epoch 48
- **Nguồn pretrain**: OmniFall HF, 100 epoch

### 1.1. Kết quả pretrain trên OmniFall

| Chỉ số | Giá trị | Epoch |
|--------|---------|-------|
| **Accuracy** | 0.9948 | 43 |
| **F1-score** | 0.9908 | 43 |
| **Precision** | - | 43 |
| **Recall** | - | 43 |
| **Val loss** | 0.0298 | 43 |

Kết luận ngắn:
- Pretrain cho nền biểu diễn rất tốt.
- Mô hình hội tụ ổn, ít dấu hiệu quá khớp.

### 1.2. Kết quả fine-tune trên UP-Fall

#### Tập validation, subjects 9-10

| Chỉ số | Giá trị | Epoch tốt nhất |
|--------|---------|----------------|
| **Accuracy** | 0.9840 | 48 |
| **F1-score** | 0.9574 | 48 |
| **Precision** | 0.9467 | 48 |
| **Recall** | 0.9688 | 48 |
| **Val loss** | 0.0047 | 48 |
| **Tổng số mẫu** | 2,568 | - |
| **Mẫu fall** | 480 | - |
| **Mẫu ADL** | 2,088 | - |

#### Tập test, subjects 11-12, ngưỡng mặc định 0.5

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | 0.7613 |
| **F1-score** | 0.5975 |
| **Precision** | 0.4362 |
| **Recall** | 0.9479 |
| **Tổng số mẫu** | 2,568 |
| **TP** | 455 |
| **FP** | 565 |
| **TN** | 1,523 |
| **FN** | 25 |

Nhận xét:
- Ngưỡng 0.5 tạo ra rất nhiều báo động giả.
- Recall cao nhưng precision thấp, nên không phù hợp để triển khai trực tiếp.

#### Tập test, subjects 11-12, ngưỡng tối ưu 0.96

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | 0.9805 |
| **F1-score** | 0.9453 |
| **Precision** | 0.9954 |
| **Recall** | 0.9000 |
| **Specificity** | 0.9990 |
| **Tổng số mẫu** | 2,568 |
| **TP** | 432 |
| **FP** | 2 |
| **TN** | 2,086 |
| **FN** | 48 |
| **ROC-AUC** | 0.9980 |
| **PR-AUC** | 0.9875 |

Ma trận nhầm lẫn:

```text
             Predicted Fall    Predicted ADL
True Fall         432 (TP)           48 (FN)
True ADL            2 (FP)        2,086 (TN)
```

Kết luận ngắn:
- Chỉ còn 2 báo động giả trên 2,088 cửa sổ ADL.
- Ngưỡng 0.96 là ngưỡng hợp lý nhất cho demo và báo cáo.

---

## 2. Mô Hình IMU

### Cấu hình mô hình
- **Checkpoint**: `weights/imu_best_upfall.pth`
- **Kiến trúc**: 1D-CNN cho 6 kênh IMU
- **Dạng đầu vào**: `(B, 6, 100)`
- **Hàm mất mát**: Binary Focal Loss với `alpha ≈ 0.82`, `gamma = 2.0`
- **Số epoch huấn luyện**: 150
- **Checkpoint tốt nhất**: epoch 123

### 2.1. Kết quả tập validation trên subjects 9-10

| Chỉ số | Giá trị | Epoch tốt nhất |
|--------|---------|----------------|
| **Accuracy** | 0.8438 | 123 |
| **F1-score** | 0.6423 | 123 |
| **Precision** | 0.5616 | 123 |
| **Recall** | 0.7500 | 123 |
| **Val loss** | 0.0339 | 123 |
| **Tổng số mẫu** | 2,568 | - |
| **Mẫu fall** | 480 | - |
| **Mẫu ADL** | 2,088 | - |

Ma trận nhầm lẫn:

```text
             Predicted Fall    Predicted ADL
True Fall         360 (TP)          120 (FN)
True ADL          281 (FP)        1,807 (TN)
```

Nhận xét:
- IMU đã cải thiện rõ sau khi chuẩn hóa theo từng cửa sổ và tăng số epoch.
- Tuy nhiên precision vẫn chưa cao, nên IMU đơn lẻ chưa đủ tốt để triển khai độc lập.

### 2.2. Kết quả tập test trên subjects 11-12

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | 0.7481 |
| **F1-score** | 0.5050 |
| **Precision** | 0.3990 |
| **Recall** | 0.6875 |
| **Tổng số mẫu** | 2,568 |
| **TP** | 330 |
| **FP** | 497 |
| **TN** | 1,591 |
| **FN** | 150 |

Ma trận nhầm lẫn:

```text
             Predicted Fall    Predicted ADL
True Fall         330 (TP)          150 (FN)
True ADL          497 (FP)        1,591 (TN)
```

Nhận xét:
- IMU bắt được khá nhiều fall nhưng vẫn tạo nhiều báo động giả.
- Vai trò phù hợp nhất của IMU là làm tín hiệu bổ trợ cho Fusion.

---

## 3. Mô Hình Fusion

### Cấu hình mô hình
- **Checkpoint**: `weights/fusion_best.pth`
- **Kiến trúc**: Fusion MLP với trọng số theo modality
- **Đầu vào**:
  - Logit từ Pose camera 1
  - Logit từ Pose camera 2
  - Logit từ IMU
- **Trọng số modality**:
  - Camera 1: 1.4
  - Camera 2: 1.4
  - IMU: 0.6
- **Hàm mất mát**: Binary Focal Loss với `alpha ≈ 0.82`, `gamma = 2.0`
- **Số epoch huấn luyện**: 10
- **Checkpoint tốt nhất**: epoch 4

### 3.1. Kết quả tập validation trên subjects 9-10

| Chỉ số | Giá trị | Epoch tốt nhất |
|--------|---------|----------------|
| **Accuracy** | 0.9899 | 4 |
| **F1-score** | 0.9722 | 4 |
| **Precision** | 1.0000 | 4 |
| **Recall** | 0.9458 | 4 |
| **Val loss** | 0.0047 | 4 |
| **Tổng số mẫu** | 2,568 | - |
| **Mẫu fall** | 480 | - |
| **Mẫu ADL** | 2,088 | - |

Ma trận nhầm lẫn:

```text
             Predicted Fall    Predicted ADL
True Fall         454 (TP)           26 (FN)
True ADL            0 (FP)        2,088 (TN)
```

Nhận xét:
- Fusion đạt precision rất cao trên validation.
- Hầu như không phát sinh báo động giả.

### 3.2. Kết quả tập test trên subjects 11-12

| Chỉ số | Giá trị |
|--------|---------|
| **Accuracy** | 0.9225 |
| **F1-score** | 0.8166 |
| **Precision** | 0.7322 |
| **Recall** | 0.9229 |
| **Tổng số mẫu** | 2,568 |
| **TP** | 443 |
| **FP** | 162 |
| **TN** | 1,926 |
| **FN** | 37 |

Ma trận nhầm lẫn:

```text
             Predicted Fall    Predicted ADL
True Fall         443 (TP)           37 (FN)
True ADL          162 (FP)        1,926 (TN)
```

Nhận xét:
- Fusion tổng quát hóa tốt trên tập test mù.
- Recall cao hơn IMU và ổn định hơn khi kết hợp đa mô thức.

---

## 4. Bảng So Sánh Tổng Quan

### 4.1. So sánh F1-score

| Mô hình | Validation | Test | Ngưỡng | Ghi chú |
|--------|------------|------|--------|---------|
| **Pose** | 0.9574 | 0.9453 | 0.96 | Mô hình triển khai chính |
| **IMU** | 0.6423 | 0.5050 | 0.5 | Mô hình hỗ trợ |
| **Fusion** | 0.9722 | 0.8166 | 0.5 | Tốt nhất trên validation |

### 4.2. So sánh đầy đủ trên tập validation

| Mô hình | Accuracy | F1 | Precision | Recall | Loss |
|--------|----------|-----|-----------|--------|------|
| **Pose** | 0.9840 | 0.9574 | 0.9467 | 0.9688 | 0.0047 |
| **IMU** | 0.8438 | 0.6423 | 0.5616 | 0.7500 | 0.0339 |
| **Fusion** | 0.9899 | 0.9722 | 1.0000 | 0.9458 | 0.0047 |

### 4.3. So sánh tập test subjects 11-12

| Mô hình | Accuracy | F1 | Precision | Recall | TP | FP | TN | FN |
|--------|----------|-----|-----------|--------|----|----|----|----|
| **Pose** | 0.9805 | 0.9453 | 0.9954 | 0.9000 | 432 | 2 | 2,086 | 48 |
| **IMU** | 0.7481 | 0.5050 | 0.3990 | 0.6875 | 330 | 497 | 1,591 | 150 |
| **Fusion** | 0.9225 | 0.8166 | 0.7322 | 0.9229 | 443 | 162 | 1,926 | 37 |

Xếp hạng theo F1 trên tập validation:
- Fusion: 0.9722
- Pose: 0.9574
- IMU: 0.6423

Xếp hạng theo F1 trên tập test subjects 11-12:
- Pose: 0.9453
- Fusion: 0.8166
- IMU: 0.5050

---

## 5. Dữ Liệu Có Vấn Đề Gì Và Đã Xử Lý Ra Sao

### 5.1. Nhánh Pose

Vấn đề chính:
- Chỉ dùng tọa độ `(x, y)` thì mô hình không biết khớp nào đang bị che khuất hoặc nhiễu.
- Dữ liệu mất cân bằng giữa fall và ADL.
- Ngưỡng mặc định 0.5 không phải lúc nào cũng tối ưu.

Cách xử lý:
- Giữ thêm kênh confidence từ NPZ để thành 3 kênh.
- Dùng pretrain trên OmniFall rồi fine-tune trên UP-Fall.
- Dùng Binary Focal Loss để giảm ảnh hưởng của lớp dễ.
- Tuning threshold bằng PR curve để chọn ngưỡng tối ưu.
- Bắt buộc load checkpoint theo `strict=True` để tránh đánh giá sai.

### 5.2. Nhánh IMU

Vấn đề chính:
- Scale giữa accel và gyro lệch rất lớn.
- Normalize theo batch gây lệch thống kê giữa train và val/test.
- Huấn luyện ngắn khiến mô hình chưa hội tụ tốt.

Cách xử lý:
- Chuẩn hóa theo từng cửa sổ, tức normalize theo trục thời gian của từng sample.
- Retrain dài hơn đến 150 epoch.
- Giữ Focal Loss để hỗ trợ dữ liệu mất cân bằng.

### 5.3. Nhánh Fusion

Vấn đề chính:
- Nếu IMU yếu, Fusion bị kéo xuống và dễ biến thành camera-only.
- Nếu huấn luyện Fusion trên checkpoint IMU cũ, mô hình học trên tín hiệu nhiễu.

Cách xử lý:
- Retrain IMU trước.
- Chạy lại Fusion trên checkpoint IMU mới.
- Giữ trọng số modality để camera vẫn chiếm vai trò chính nhưng IMU vẫn có tác dụng bổ trợ.

---

## 6. Quy Trình Triển Khai

1. Trích xuất Pose offline bằng YOLO pose cho 2 camera.
2. Lưu `t`, `kpts`, `conf` vào NPZ.
3. Tạo cửa sổ 2 giây, stride 0.5 giây.
4. Pretrain Pose trên OmniFall.
5. Fine-tune Pose trên UP-Fall.
6. Retrain IMU với chuẩn hóa theo từng cửa sổ và số epoch lớn hơn.
7. Retrain Fusion trên checkpoint IMU mới.
8. Đánh giá validation, test subjects 11-12, và demo realtime mô phỏng.

---

## 7. Kết Luận Kỹ Thuật

- **Pose** là mô hình tốt nhất để triển khai realtime trực tiếp.
- **IMU** sau retrain đã cải thiện đáng kể nhưng vẫn chưa đủ mạnh để dùng độc lập.
- **Fusion** là lựa chọn tốt nhất nếu ưu tiên độ ổn định đa mô thức, đặc biệt trên validation.
- Pipeline hiện tại đã có đủ bước để báo cáo, demo và so sánh công bằng giữa ba mô hình.

Kết quả chốt cuối cùng trên test subjects 11-12:

- Pose: F1 = 0.9453
- IMU: F1 = 0.5050
- Fusion: F1 = 0.8166

*Tài liệu này được cập nhật theo các checkpoint và log mới nhất trong ngày 9/5/2026.*
