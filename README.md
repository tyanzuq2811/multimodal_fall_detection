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
  - Tensor: (B, 2, 60, 17)
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
