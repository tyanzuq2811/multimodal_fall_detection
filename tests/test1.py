import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFallDetector(nn.Module):
    def __init__(self, num_joints=17, sensor_dim=3, num_classes=2):
        super(MultimodalFallDetector, self).__init__()
        self.sensor_dim = sensor_dim
        
        # Nhánh 1: Xử lý Khung xương (Tọa độ X,Y từ YOLO/MediaPipe)
        # Giả sử đầu vào là (Batch, Time, Joints, Channels=2)
        self.skeleton_branch = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(3, 3), padding=(1, 1)), # Spatial
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Pooling -> (B, 64, 1, 1)
        )
        
        # Nhánh 2: Xử lý Gia tốc (3 trục XYZ từ thiết bị IoT)
        # Sử dụng Bi-LSTM kết hợp Channel Attention 
        self.sensor_branch = nn.LSTM(
            input_size=sensor_dim, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        self.sensor_attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Lớp Dung hợp Đặc trưng (Fusion Layer)
        # Gộp (Concatenate) vector 64-dim (Video) và 256-dim (Sensor)
        self.fusion_layer = nn.Linear(64 + 256, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x_skeleton, x_sensor):
        # 1. Trích xuất đặc trưng Video
        # Kỳ vọng x_skeleton shape: (B, T, J, 2) hoặc (B, 2, T, J)
        if x_skeleton.dim() == 4 and x_skeleton.size(-1) == 2:
            x_skeleton = x_skeleton.permute(0, 3, 1, 2).contiguous()  # (B, 2, T, J)
        elif x_skeleton.dim() == 4 and x_skeleton.size(1) == 2:
            x_skeleton = x_skeleton.contiguous()  # (B, 2, T, J)
        else:
            raise ValueError(
                f"x_skeleton phải có shape (B, T, J, 2) hoặc (B, 2, T, J), nhưng nhận {tuple(x_skeleton.shape)}"
            )
        video_feat = self.skeleton_branch(x_skeleton)
        video_feat = video_feat.flatten(1)  # (B, 64)
        
        # 2. Trích xuất đặc trưng Sensor
        # Kỳ vọng x_sensor shape: (B, T, sensor_dim)
        if x_sensor.dim() != 3 or x_sensor.size(-1) != self.sensor_dim:
            raise ValueError(
                f"x_sensor phải có shape (B, T, {self.sensor_dim}), nhưng nhận {tuple(x_sensor.shape)}"
            )
        sensor_out, _ = self.sensor_branch(x_sensor)
        # Áp dụng Attention để tìm khoảnh khắc va chạm mạnh nhất 
        attn_weights = self.sensor_attention(sensor_out)
        sensor_feat = torch.sum(sensor_out * attn_weights, dim=1)
        
        # 3. Dung hợp (Feature Concatenation) 
        combined = torch.cat((video_feat, sensor_feat), dim=1)
        
        # 4. Phân loại
        x = F.relu(self.fusion_layer(combined))
        return self.classifier(x)

# Khởi tạo model
model = MultimodalFallDetector()
print("Sẵn sàng huấn luyện đa phương thức!")