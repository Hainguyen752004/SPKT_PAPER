# SkinSeg-YOLO26-P2Attn

Mã nguồn thử nghiệm cho **phân đoạn thể hiện (instance segmentation) tổn thương da** với hướng nghiên cứu: đánh giá độ bền trước artifact và attention trên đặc trưng độ phân giải cao. Đây là giả thuyết cần kiểm chứng, không phải kết quả đã được xác nhận.

Tên mô hình dùng trong dự án là **SkinSeg-YOLO26-P2Attn** (tên triển khai: YOLO26n P2-CBAM). Không gọi tổn thương là “tế bào”; dữ liệu và nhãn biểu diễn vùng tổn thương da trên ảnh dermoscopic.

## Kiến trúc

`models/yolo26n-seg-p2-cbam.yaml` dựa trên YOLO26n thực, gồm các khối `C3k2`, `C2PSA` và đầu phân đoạn `Segment26`. YOLO26 chính thức đã có topology phát hiện P2; dự án thích nghi topology đó thành đầu **phân đoạn bốn mức P2/P3/P4/P5** và đặt CBAM tại các đặc trưng độ phân giải cao P2 và P3.

- P2 có stride 4 (feature map 160 × 160 với đầu vào 640 × 640), nhằm giữ thông tin biên và tổn thương nhỏ.
- P3/P4/P5 bổ sung ngữ cảnh đa tỉ lệ.
- CBAM lần lượt áp dụng channel attention và spatial attention, không đổi kích thước tensor.

P2, CBAM và kỹ thuật loại lông lấy cảm hứng từ DullRazor đều dựa trên ý tưởng đã tồn tại; đóng góp cần được trình bày là cách tích hợp và đánh giá chúng trong bài toán cụ thể, không phải tính mới riêng lẻ của từng thành phần.

## Dữ liệu và provenance

Nguồn đầu vào bất biến của pipeline:

`data/dataset_yolo_fixed_labels/dataset_yolo`

Số lượng nguồn đã biết là **train 8.008, val 998, test 1.007** cặp ảnh–nhãn. Hai thư mục dữ liệu đã xử lý được phân tích trong dự án là:

- `data/dataset_yolo_640x640_multiview`
- `data/dataset_yolo_aug_p2_cbam`

Quan sát hiện có trên các thư mục này: multiview train **16.016**; augmented train **31.880**; validation **998**; test **1.007**. Đây là số lượng file quan sát được, không phải bằng chứng tự thân về cân bằng lớp hay không rò rỉ dữ liệu.

Pipeline giữ nguyên split theo thiết kế: preprocessing và augmentation chỉ ghi biến thể vào split tương ứng; augmentation chỉ chạy trên train. Công cụ audit hiện chỉ chuẩn hóa **ISIC ID ở mức ảnh** (bỏ hậu tố `_v1`, `_v7`, `_aug*`) và kiểm tra giao nhau giữa split. Kiểm tra này không chứng minh không rò rỉ ở mức bệnh nhân. Trước khi công bố cần có metadata bệnh nhân/ca bệnh và audit split được nhóm theo bệnh nhân.

## Artifact-oriented fixed multi-view dermoscopic preprocessing

Tên chính xác của giai đoạn 1 là **artifact-oriented fixed multi-view dermoscopic preprocessing** (tiền xử lý ảnh soi da đa phiên bản cố định, định hướng xử lý artifact). Đây là một chuỗi cố định áp dụng cho **mọi ảnh train**, không phải bộ phát hiện artifact có điều kiện. Giai đoạn này xác định (deterministic) và tách biệt với augmentation ngẫu nhiên ở giai đoạn 2. `v1` và `v7` chỉ là định danh kế thừa (legacy identifiers), không biểu thị bảy phiên bản.

- **Train v1:** ảnh nguồn → letterbox 640 × 640.
- **Train v7 (artifact-processed view):** hàm `smart_roi_crop` dùng ngưỡng xám cố định 15, lấy external contour lớn nhất rồi crop bounding box để xử lý ROI/viền tối; bước này không “thông minh” hay phát hiện artifact theo học máy dù tên hàm là `smart_roi_crop`. Sau đó là **DullRazor-inspired hair removal**: morphological black-hat với kernel chữ thập 17 × 17, threshold nhị phân cố định 10, rồi `cv2.inpaint` bán kính 3 với `INPAINT_TELEA`; tiếp theo Gray-World color constancy và letterbox 640 × 640. Cách triển khai này lấy cảm hứng từ DullRazor, không được khẳng định tương đương thuật toán DullRazor chuẩn.
- **Val/Test:** chỉ tạo v1 bằng letterbox; không tạo v7 và không augmentation.

Nhãn polygon được biến đổi đồng bộ với crop và letterbox.

## NV-excluding image-level augmentation

Giai đoạn 2 sao chép dữ liệu đã preprocessing rồi áp dụng **NV-excluding image-level augmentation**: mọi ảnh có ít nhất một polygon class 5 (NV) đều bị bỏ qua; mọi ảnh không chứa NV được xử lý như nhau, không đặt tần suất đích theo lớp. Mỗi ảnh đủ điều kiện có tối đa **3 lần thử**. Một lần thử không hợp lệ có thể không sinh file, vì vậy đây không phải thuật toán cân bằng lớp được bảo đảm.

Các phép biến đổi đúng theo mã:

| Biến đổi | Tham số | Xác suất |
|---|---|---:|
| Horizontal flip | — | 0,50 |
| Vertical flip | — | 0,25 |
| Affine | scale 0,85–1,15; translate x/y ±5%; rotate ±30° | 0,90 |
| Affine shear | x/y ±10° | 0,35 |
| Brightness/contrast | limit 0,12/0,12 | 0,50 |
| Hue/saturation/value | 5/8/5 | 0,25 |
| Gaussian blur | kernel 3–5 | 0,15 |

Các đỉnh polygon được biến đổi bằng cùng phép biến đổi ảnh, chuẩn hóa lại về [0,1]; polygon có ít hơn 3 điểm hoặc diện tích **< 50 px²** bị loại. Nếu không còn polygon hợp lệ, lần thử không được lưu.

## Tái lập

Môi trường tham chiếu yêu cầu `ultralytics==8.4.60` để có `Segment26`. Các dependency cốt lõi khác gồm PyTorch, PyYAML, OpenCV, NumPy, tqdm và Albumentations:

```bash
python -m pip install ultralytics==8.4.60 torch PyYAML opencv-python numpy tqdm albumentations
```

Đặt provenance cục bộ tại đúng đường dẫn `data/dataset_yolo_fixed_labels/dataset_yolo` với số file kỳ vọng 8.008/998/1.007 cho train/val/test. Repository hiện chưa ghi đầy đủ nguồn tải, phiên bản, checksum, giấy phép và metadata bệnh nhân của dữ liệu; đây là điều kiện tiên quyết phải bổ sung cho bài báo, không được suy đoán.

Chạy pipeline:

```bash
python data_processing/01_preprocess.py
python data_processing/02_augment.py
python 03_train_p2_cbam.py
```

Hai script dữ liệu yêu cầu thư mục đích chưa tồn tại. `--overwrite` sẽ **xóa toàn bộ đúng thư mục đích đã khóa trong script rồi tạo lại**, vì vậy chỉ dùng khi chủ động tái tạo processed data:

```bash
python data_processing/01_preprocess.py --overwrite
python data_processing/02_augment.py --overwrite
```

Augmentation hiện ngẫu nhiên và **không đặt seed**, nên tập 31.880 ảnh là quan sát của lần chạy đã phân tích, không phải số lượng bảo đảm tái lập. Muốn tái lập bit-level cần bổ sung chính sách seed và cố định phiên bản dependency.

Audit không thay đổi dữ liệu và ghi báo cáo JSON dưới `audit_reports/`:

```bash
python -m data_processing.audit_dataset --project-root . --report audit_reports/processed.json
```

Kết quả mong đợi là exit code 0 và `audit_reports/processed.json` chứa kiểm tra layout/count, cặp ảnh–nhãn, polygon, thành phần tên view, phân bố lớp và overlap ISIC ID đã chuẩn hóa ở **mức ảnh**. Đây không phải patient-level leakage audit.

Script train dùng `models/yolo26n-seg-p2-cbam.yaml`, chuyển một phần trọng số tương thích từ checkpoint YOLO26n-seg, và ghi kết quả vào `runs/segment`.

## Trạng thái bằng chứng

Repository chưa cung cấp đủ kết quả đã kiểm chứng để khẳng định vượt SOTA, tốc độ FPS cố định, hoặc các giá trị mAP/Dice cụ thể. Những số liệu đó phải được đo trên cùng split, phần cứng và protocol rồi báo cáo kèm độ bất định. Kế hoạch baseline/ablation được nêu trong `tailieu.md`.
## Quick training path

On the current Windows machine, prefer the environment interpreter directly instead of `conda run`, because the base
conda wrapper can fail while printing Unicode training logs to a CP1252 console.

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 5 --fraction 0.10 --batch 8 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5
```

The train script supports fast-run controls:

- `--fraction`: train-data fraction in `(0, 1]`;
- `--batch`: batch-size override;
- `--workers`: dataloader worker override;
- `--name`: Ultralytics run-name override.

The P2-CBAM model registers `P2CompatibleSegment26` under the `Segment26` parser name. It keeps P2-P5 prediction
features, but uses P3-P5 for the prototype branch so the segmentation validator receives stride-4 masks.

Latest quick sanity result, not a paper-final number:

- run: `runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5`;
- train: 10% of `dataset_yolo_aug_p2_cbam`, 5 epochs, batch 8;
- validation: 998 images;
- final all-class box mAP50/mAP50-95: `0.169` / `0.107`;
- final all-class mask mAP50/mAP50-95: `0.163` / `0.118`.
