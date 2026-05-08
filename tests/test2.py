try:
	from datasets import load_dataset
except ImportError as e:
	raise SystemExit(
		"Thiếu thư viện 'datasets'. Cài tối thiểu bằng: pip install datasets"
	) from e

# Nạp bộ dữ liệu OmniFall (chứa cả CMDFall và UP-Fall đã đồng bộ nhãn)
# Config 'of-sta-cs' là tập dữ liệu dàn dựng, chia theo Subject 
dataset = load_dataset("simplexsigil2/omnifall", "of-sta-cs")

# Xem thông tin dữ liệu nạp được
train_ds = dataset["train"]
first = train_ds[0]

print(f"Số lượng mẫu huấn luyện: {len(train_ds)}")
print(f"Các cột trong train: {train_ds.column_names}")
print(f"Nhãn mẫu đầu tiên: {first.get('label')}")
print(f"Path mẫu đầu tiên: {first.get('path')}")

label_feature = train_ds.features.get("label")
label_names = getattr(label_feature, "names", None)
if label_names:
	print(f"Mapping label -> tên lớp: {label_names}")

# Lưu ý: 'path' trong dataset sẽ trỏ đến file video/csv tương ứng
# Bạn có thể dùng OpenCV nạp video và Pandas nạp CSV theo đường dẫn này.