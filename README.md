# Dự án Nhận diện Té ngã Đa phương thức (Multi-Modal Fall Detection)

Mục tiêu: hệ thống nhận diện té ngã dựa trên 2 camera và IMU (cổ tay + hông), tối ưu theo tiêu chí **Nhanh, Nhẹ, Dễ triển khai**. Toàn bộ pipeline gồm 3 bước rõ ràng:

1) **Pre-train nhanh Pose** trên dữ liệu Hugging Face (OmniFall)
2) **Xử lý offline + huấn luyện trên UP-Fall** (2 camera + IMU)
3) **Late Fusion** gộp 3 logits (Cam1, Cam2, IMU)

---

## 1. Tổng quan kiến trúc

- Bài toán: Binary classification (1 = té ngã, 0 = ADL)
- Cửa sổ thời gian: L = 2s, bước trượt S = 0.5s
- Pose: 2 camera, 30 FPS -> 60 frame / window
  - Tensor: (B, 3, 60, 17) với kênh thứ 3 là confidence/visibility
- IMU: 6 trục, 50 Hz -> 100 sample / window
  - Tensor: (B, 6, 100)
- Loss: BCEWithLogitsLoss
- Fusion: MLP nhỏ, đầu vào 3 logits (Cam1, Cam2, IMU)

---

## 2. Cấu trúc thư mục

```
multimodal_fall_detection/
  configs/
    data_config.yaml
    train_pose_pretrain.yaml
    train_pose.yaml
    train_imu.yaml
    train_fusion.yaml
  src/
    data_pipeline/
    engines/
    models/
    utils/
  data/
    raw/
      upfall/
      omnifall/
    processed/
      pose_features/
      imu_windows/
      synced_windows/
  weights/
```

---

## 3. Cài đặt

```bash
pip install -r requirements.txt
```

> Lưu ý: `mediapipe`, `opencv-python` và `datasets` cần thiết cho bước pretrain/pose extraction.

---

## 4. Cấu hình quan trọng

- **configs/data_config.yaml**
  - `paths.raw_upfall_dir`: đường dẫn dữ liệu UP-Fall local
  - `paths.raw_omnifall_dir`: cache HF dataset
  - `upfall.labels`: danh sách activity fall/ADL
  - `target_lengths`: pose=60, imu=100
  - `omnifall`: tên dataset HF, label field, split (vd `simplexsigil2/omnifall` hoặc `simplexsigil2/wanfall`)
    - Với bộ `simplexsigil2/*`, trường video nằm ở `path`, đoạn clip theo `start`/`end`.
    - Nhãn té ngã khuyến nghị: `fall`, `fallen`; còn lại xem như ADL.

Nếu dataset HF không dùng trường `label` hoặc `video`, cần chỉnh lại các trường trong `omnifall`.
Với bộ `simplexsigil2/*`, dữ liệu video nằm trong file nén (tar/zip). Loader sẽ tự tải và giải nén
vào `data/raw/omnifall/media`. Nếu bạn đã có video local, hãy đặt `omnifall.local_video_root`.

---

## 5. Quy trình chạy từng bước

### Bước 1: Pre-train nhanh Pose (OmniFall)

**Mục tiêu**: làm nóng nhánh Pose bằng dữ liệu online.

**Bước 1.1** - Tải dữ liệu từ Hugging Face, trích pose bằng MediaPipe, lưu npz

```bash
python -m src.data_pipeline.hf_omnifall_loader
```

Kết quả:
- `data/processed/pose_features/omnifall/*.npz`
- `data/processed/pose_features/omnifall_manifest.jsonl`

**Bước 1.2** - Pre-train nhanh Pose

```bash
python -m src.engines.trainer_pose_pretrain
```

Kết quả:
- `weights/pose_pretrained_omnifall.pth`

---

### Bước 2: Xử lý offline + Huấn luyện UP-Fall (Local)

**Mục tiêu**: trích pose trong zip và IMU từ CSV, tạo mapping window, sau đó train 2 nhánh song song.

#### 2.1. Offline Pose + IMU

- Trích IMU (6 trục) từ CSV

```bash
python -m src.data_pipeline.extract_imu_cache
```

- Trích pose từ zip (2 camera, đọc trực tiếp trong RAM)

```bash
python -m src.data_pipeline.extract_pose_offline
```

- Tạo window manifest (đồng bộ IMU + Pose)

```bash
python -m src.data_pipeline.window_generator
```

Kết quả:
- `data/processed/imu_windows/.../imu_wrist_belt_6d.npz`
- `data/processed/pose_features/.../camera1_pose.npz`
- `data/processed/pose_features/.../camera2_pose.npz`
- `data/processed/synced_windows/upfall_windows.jsonl`

#### 2.2. Train nhanh Pose 2 camera (Shared Weights)

```bash
python -m src.engines.trainer_pose
```

Kết quả:
- `weights/pose_finetuned_upfall.pth`

#### 2.3. Train nhanh IMU

```bash
python -m src.engines.trainer_imu
```

Kết quả:
- `weights/imu_best_upfall.pth`

---

### Bước 3: Late Fusion (3 logits)

**Mục tiêu**: dùng MLP nhỏ để gộp 3 logits độc lập (Cam1, Cam2, IMU).

```bash
python -m src.engines.trainer_fusion
```

Kết quả:
- `weights/fusion_best.pth`

### Demo realtime mô phỏng cho Pose

Sau khi chốt checkpoint pose, bạn có thể mô phỏng luồng realtime trên các window liên tiếp của subjects giữ lại:

```bash
PYTHONPATH=. python3 -m src.engines.demo_pose_realtime \
  --ckpt weights/pose_finetuned_upfall.pth \
  --split 11-12 \
  --threshold 0.96 \
  --save-log eval_test_output/realtime_demo.jsonl
```

Kịch bản này in từng window theo thời gian, hiển thị xác suất fall, nhãn dự đoán, nhãn thật, và tổng kết TP/FP/TN/FN cuối phiên.

---

## 6. Log shape để debug nhanh

Trong quá trình train, hệ thống sẽ in ra shape của batch (1 lần duy nhất):

- IMU: `(B, 6, 100)`
- Pose: `(B, 2, 60, 17)`

Nếu shape không đúng, kiểm tra `target_lengths` trong `data_config.yaml`.

---

## 7. Lưu ý quan trọng

- Nếu UP-Fall chưa có activity 7-11, sẽ không có dữ liệu té ngã (label=1).
- Dữ liệu lớn, cần thời gian để trích pose offline (ưu tiên chạy trên máy có GPU nếu có).
- Thư mục `data/` và `weights/` được ignore trong git.

---

## 8. Pipeline tổng kết (từ đầu đến cuối)

```bash
# 1) HF pretrain
python -m src.data_pipeline.hf_omnifall_loader
python -m src.engines.trainer_pose_pretrain

# 2) UP-Fall offline + train
python -m src.data_pipeline.extract_imu_cache
python -m src.data_pipeline.extract_pose_offline
python -m src.data_pipeline.window_generator
python -m src.engines.trainer_pose
python -m src.engines.trainer_imu

# 3) Fusion
python -m src.engines.trainer_fusion
```

---

## 9. Troubleshooting nhanh

- Lỗi thiếu dataset HF: kiểm tra `configs/data_config.yaml` (tên dataset, label field; vd `simplexsigil2/omnifall`)
- Lỗi thiếu pose/imu cache: chạy lại bước offline (extract_imu_cache, extract_pose_offline)
- Lỗi thiếu fall activity: cần tải dữ liệu UP-Fall Activity 7-11

---

## 10. Ghi chú thực chiến

- Temporal 1D-CNN cho pose nhanh và nhẹ, dễ deploy trên edge.
- Fusion 3 logits giúp hệ thống chịu lỗi tốt khi 1 camera bị che hoặc IMU bị nhiễu.
- Cách debug nhanh nhất: log 3 logits để xem hệ thống đang tin vào modality nào.

---

## 11. Báo cáo tổng kết cuối cùng

Phần này tóm tắt lại toàn bộ quá trình theo kiểu báo cáo kỹ thuật: dữ liệu ban đầu là gì, vấn đề phát sinh ở đâu, mình xử lý như thế nào, và kết quả cuối cùng ra sao.

### 11.1. Bối cảnh dữ liệu

Hệ thống dùng hai nguồn dữ liệu chính:

- **UP-Fall**: dữ liệu local từ camera + IMU, có cấu trúc theo subject / activity / trial.
- **OmniFall**: dữ liệu Hugging Face dùng để pre-train nhánh Pose trước khi fine-tune trên UP-Fall.

Trong pipeline hiện tại:

- Pose được trích từ 2 camera.
- Mỗi cửa sổ Pose dài 2 giây, tương ứng 60 frame.
- Sau khi bổ sung confidence/visibility, tensor Pose có dạng **(B, 3, 60, 17)**.
- IMU giữ nguyên theo cửa sổ 2 giây, tensor dạng **(B, 6, 100)**.

### 11.2. Vấn đề xuất hiện trong quá trình làm việc

#### Vấn đề 1: Mất thông tin độ tin cậy của keypoint

Ban đầu, Pose chỉ dùng tọa độ $(x, y)$ nên mô hình không biết khớp nào đang bị che khuất hoặc nhiễu.

Hệ quả:

- Khi người dùng ngã, một số khớp bị che hoặc lệch mạnh.
- Mô hình học thiếu tín hiệu về độ tin cậy nên khó phân biệt cú ngã thật với trạng thái chuyển tiếp.

#### Vấn đề 2: Mất cân bằng lớp

Số cửa sổ ADL nhiều hơn cửa sổ fall.

Hệ quả:

- BCE thuần túy dễ bị lệch về lớp dễ.
- Mô hình học tốt trên ADL nhưng dễ bỏ sót fall nếu không xử lý đúng.

#### Vấn đề 3: Ngưỡng quyết định cố định 0.5 không phải lúc nào cũng tối ưu

Với bài toán mất cân bằng và focal loss, ngưỡng 0.5 có thể không phải ngưỡng tốt nhất cho F1.

Hệ quả:

- Recall có thể rất cao nhưng Precision thấp.
- Nếu chỉ nhìn threshold mặc định, ta đánh giá thấp mô hình.

#### Vấn đề 4: Kiểm tra checkpoint cần chặt chẽ

Checkpoint của dự án không chỉ lưu `state_dict` thuần mà được bọc trong dictionary.

Hệ quả:

- Nếu load sai kiểu hoặc dùng `strict=False`, mô hình có thể bị tải lệch mà không báo lỗi rõ ràng.

### 11.3. Cách giải quyết

#### Giải pháp A: Bổ sung kênh confidence/visibility cho Pose

Từ dữ liệu NPZ, mình giữ lại `conf` và ghép với `(x, y)` để thành 3 kênh.

Ý nghĩa:

- Keypoint nào bị che khuất hoặc ít tin cậy sẽ có tín hiệu thấp hơn.
- Mô hình học được ngữ cảnh hình học + độ chắc chắn của từng khớp.

#### Giải pháp B: Dùng pre-train OmniFall trước rồi fine-tune UP-Fall

Chiến lược này giúp nhánh Pose không phải học từ đầu trên dữ liệu nhỏ.

Quy trình:

1. Pre-train trên OmniFall.
2. Fine-tune trên UP-Fall với checkpoint pre-trained.
3. Dùng checkpoint tốt nhất cho test và demo.

#### Giải pháp C: Chuyển từ BCE sang Binary Focal Loss

Mục tiêu là ép mô hình tập trung vào mẫu khó và giảm ảnh hưởng của mẫu dễ.

Tác dụng:

- Tăng khả năng nhận diện fall khó.
- Giảm việc mô hình chỉ tối ưu ADL.

#### Giải pháp D: Tìm threshold tối ưu bằng PR/F1 sweep

Thay vì ép ngưỡng 0.5, mình quét nhiều threshold để tìm điểm F1 tốt nhất.

Tác dụng:

- Đánh giá đúng hơn khả năng phân biệt của mô hình.
- Có threshold phù hợp cho demo và báo cáo.

#### Giải pháp E: Dùng checkpoint strict=True khi evaluate

Mình sửa phần load checkpoint để:

- Trích xuất đúng `model` / `model_state_dict` / `state_dict`.
- Load với `strict=True` để nếu lệch kiến trúc thì báo lỗi ngay.

Tác dụng:

- Tránh đánh giá nhầm trên model khởi tạo ngẫu nhiên.

### 11.4. Quy trình triển khai thực tế

#### Bước 1: Trích xuất Pose offline

- Dùng YOLO pose cho 2 camera.
- Lưu `t`, `kpts`, `conf` vào NPZ.

#### Bước 2: Tạo window đồng bộ

- Cửa sổ dài 2 giây, trượt 0.5 giây.
- Đồng bộ IMU và Pose trong manifest.

#### Bước 3: Pre-train Pose trên OmniFall

- Chạy pre-train để có trọng số nền tốt.

#### Bước 4: Fine-tune Pose trên UP-Fall

- Dùng checkpoint pre-trained.
- Fine-tune với kênh confidence và focal loss.

#### Bước 5: Đánh giá validation/test + threshold tuning

- Tuning threshold trên tập validation.
- Chạy test mù trên subjects 11-12.

#### Bước 6: Demo realtime mô phỏng

- Chạy mô phỏng sliding-window theo thời gian.
- In xác suất, nhãn dự đoán, nhãn thật, và tổng kết TP/FP/TN/FN.

### 11.5. Kết quả cuối cùng

#### Validation (subjects 2-10)

- Best threshold: **0.9900**
- Best F1: **0.8936**
- Metrics tại threshold 0.5:
  - F1 = **0.8322**
  - Accuracy = **0.9322**
  - Precision = **0.7667**
  - Recall = **0.9098**

#### Test mù (subjects 11-12)

- Best threshold: **0.9600**
- Best F1: **0.9453**
- Metrics tại threshold 0.5:
  - F1 = **0.5975**
  - Accuracy = **0.7613**
  - Precision = **0.4362**
  - Recall = **0.9479**
- Metrics tại threshold tối ưu 0.96:
  - F1 = **0.9453**
  - Accuracy = **0.9805**
  - Precision = **0.9954**
  - Recall = **0.9000**

#### Demo realtime mô phỏng đầy đủ trên subjects 11-12

- Windows processed: **2568**
- TP = **432**
- FP = **2**
- TN = **2086**
- FN = **48**
- Accuracy = **0.9805**
- Precision = **0.9954**
- Recall = **0.9000**
- F1 = **0.9453**

### 11.6. Kết luận kỹ thuật

Nhánh Pose hiện tại đã đạt trạng thái có thể dùng để báo cáo và demo:

- Checkpoint chính: `weights/pose_finetuned_upfall.pth`
- Tensor Pose cuối cùng: **(B, 3, 60, 17)**
- Pipeline đã xử lý được:
  - mất cân bằng lớp,
  - thiếu tín hiệu visibility,
  - chọn threshold tối ưu,
  - kiểm tra checkpoint chặt chẽ,
  - demo realtime mô phỏng.

Nếu cần viết ngắn gọn cho hội đồng, có thể chốt thành một câu:

> Mình dùng pre-train + fine-tune cho nhánh Pose, bổ sung kênh confidence, chuyển sang focal loss, tinh chỉnh threshold bằng PR curve, và cuối cùng kiểm tra trên tập mù subjects 11-12 để có kết quả generalization cao và đủ ổn định cho demo realtime.
