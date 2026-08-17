# Định hướng bài báo SkinSeg-YOLO26-P2Attn

## Tiêu đề đề xuất

**High-Resolution Attention-Guided YOLO26 for Skin Lesion Instance Segmentation with Multi-View Dermoscopic Preprocessing**

Tiêu đề trung tính này mô tả phương pháp mà không khẳng định trước độ bền trước artifact. Tên ngắn của hệ thống: **SkinSeg-YOLO26-P2Attn**.

## Phạm vi tuyên bố khoa học

Mô hình dùng YOLO26n thực với `C3k2`, `C2PSA`, `Segment26` và bốn mức P2/P3/P4/P5. YOLO26 chính thức đã có topology phát hiện P2; công việc này thích nghi topology đó cho phân đoạn bốn mức và chèn CBAM tại P2/P3. Không tuyên bố P2, CBAM hoặc bước loại lông lấy cảm hứng từ DullRazor là phát minh riêng.

Đóng góp có thể bảo vệ sau khi có thực nghiệm:

1. Một pipeline **artifact-oriented fixed multi-view dermoscopic preprocessing** (tiền xử lý ảnh soi da đa phiên bản cố định, định hướng xử lý artifact) tạo hai phiên bản train bổ trợ. Chuỗi cố định chạy trên mọi ảnh train, không phát hiện artifact có điều kiện. `v1`/`v7` là tên kế thừa: v1 letterbox; v7 là artifact-processed view gồm crop theo threshold/contour, loại lông lấy cảm hứng từ DullRazor, Gray-World và letterbox. Val/test chỉ dùng v1.
2. Một cấu hình YOLO26n phân đoạn P2/P3/P4/P5 với CBAM trên đặc trưng độ phân giải cao.
3. Đánh giá có kiểm soát giả thuyết về artifact robustness, tổn thương nhỏ/biên khó, và **NV-excluding image-level augmentation** trên cùng split.

Đây là giả thuyết nghiên cứu cho tới khi thí nghiệm xác nhận. Không dùng các từ “first”, “100%”, “guaranteed balance”, “SOTA superiority”, hoặc số FPS/mAP/Dice chưa đo. Không gọi tổn thương da là “tế bào ung thư”.

## Dữ liệu và kiểm soát rò rỉ

Provenance đầu vào: `data/dataset_yolo_fixed_labels/dataset_yolo`, với số lượng đã biết train/val/test lần lượt **8.008/998/1.007**.

Hai thư mục processed đã được phân tích là `data/dataset_yolo_640x640_multiview` và `data/dataset_yolo_aug_p2_cbam`. Quan sát hiện có: multiview train 16.016; augmented train 31.880; val 998; test 1.007.

Thiết kế bảo toàn split: chỉ train có v7 và augmentation; val/test chỉ có v1 letterbox. Audit hiện tại chỉ ánh xạ tên biến thể về **ISIC ID mức ảnh** và kiểm tra giao giữa ba split; nó không kiểm tra patient-level leakage. Trước công bố phải có metadata bệnh nhân, tạo split theo nhóm bệnh nhân và audit lại. Số file hay normalized image-ID overlap bằng không không chứng minh zero leakage ở mức bệnh nhân.

Giai đoạn augmentation tách biệt với preprocessing cố định. Đây là **NV-excluding image-level augmentation**: ảnh có class 5 NV bị bỏ qua; mọi ảnh không chứa NV được xử lý như nhau, không có target frequency theo lớp. Mỗi ảnh đủ điều kiện có tối đa ba lần thử; HFlip 50%, VFlip 25%, affine scale 0,85–1,15 + translate ±5% + rotate ±30° ở 90%, shear ±10° ở 35%, brightness/contrast 0,12 ở 50%, HSV 5/8/5 ở 25%, Gaussian blur 3–5 ở 15%. Polygon được biến đổi đồng bộ; polygon dưới 50 px² hoặc dưới ba điểm bị loại, và lần thử rỗng không được lưu. Quá trình hiện stochastic, không đặt seed; 31.880 chỉ là số file quan sát, không phải kết quả chắc chắn tái lập.

Triển khai v7 dùng `smart_roi_crop`, nhưng tên hàm không hàm ý nhận biết thông minh: grayscale threshold cố định 15, external contour lớn nhất và bounding-box crop. Bước **DullRazor-inspired hair removal** dùng black-hat kernel chữ thập 17 × 17, binary threshold 10 và Telea inpainting bán kính 3. Không khẳng định triển khai này tương đương DullRazor chuẩn.

## Bố cục bài báo

1. **Abstract:** bài toán, phương pháp, protocol, và chỉ điền kết quả đã đo.
2. **Introduction:** artifact dermoscopic, mất chi tiết ở biên/tổn thương nhỏ, mất cân bằng lớp; nêu đóng góp ở mức tích hợp và đánh giá.
3. **Related Work:** skin-lesion segmentation, DullRazor-inspired artifact handling, multi-scale YOLO segmentation, attention modules.
4. **Method:** provenance và split; preprocessing v1/v7; NV-excluding image-level augmentation (tăng cường dữ liệu mức ảnh loại trừ NV); YOLO26n (`C3k2`, `C2PSA`, `Segment26`); nhánh P2–P5 và vị trí CBAM.
5. **Experimental Protocol:** baseline, ablation, seed, phần cứng, hyperparameters, metric và khoảng tin cậy; audit ID.
6. **Results:** chất lượng tổng thể, theo lớp, theo kích thước tổn thương, robustness theo artifact, latency/throughput đo thực tế.
7. **Discussion:** hiệu ứng từng thành phần, failure cases, chi phí tính toán, giới hạn dữ liệu và khả năng khái quát.
8. **Conclusion:** kết luận chỉ trong phạm vi bằng chứng.

## Ma trận thí nghiệm tối thiểu

| ID | Backbone/head | P2–P5 segmentation | CBAM P2/P3 | v7 artifact view | NV-excluding aug | Mục tiêu |
|---|---|---:|---:|---:|---:|---|
| B0 | YOLO26n-seg chính thức | Theo baseline | Không | Không | Không | Baseline chính |
| B1 | Custom YOLO26n | Có | Không | Không | Không | Hiệu ứng thích nghi 4-scale segmentation |
| B2 | Custom YOLO26n | Có | Có | Không | Không | Hiệu ứng CBAM |
| B3 | Custom YOLO26n | Có | Không | Có | Không | Hiệu ứng fixed artifact-oriented multiview |
| B4 | Custom YOLO26n | Có | Không | Không | Có | Hiệu ứng NV-excluding image-level augmentation |
| P | Custom YOLO26n | Có | Có | Có | Có | Hệ thống đầy đủ |

Nên thêm factorial ablation nếu ngân sách cho phép để phát hiện tương tác giữa P2/P3 CBAM, v7 và augmentation. Mọi cấu hình phải dùng cùng source split, seed set, epoch budget và quy tắc chọn checkpoint.

## Chỉ số và báo cáo cần có

- Mask mAP50 và mAP50–95, precision, recall; Dice/IoU nếu pipeline đánh giá định nghĩa rõ cách tổng hợp.
- Kết quả macro và theo từng lớp, không chỉ micro/overall.
- Phân tầng theo diện tích tổn thương và mức artifact; cần định nghĩa hoặc gán nhãn artifact trước.
- Mean ± standard deviation hoặc khoảng tin cậy qua nhiều seed.
- Params, FLOPs, peak memory, latency và FPS được đo trên phần cứng/batch cụ thể, có warm-up.
- Kiểm định thống kê hoặc bootstrap paired trên cùng test cases khi so sánh mô hình.
- Failure cases và định tính attention chỉ là bằng chứng hỗ trợ, không thay thế metric. Grad-CAM/attention map không được mô tả bằng tỷ lệ tập trung nếu chưa có phép đo định lượng.

## Kết quả còn thiếu trước khi viết claim

- Audit hiện tại về normalized ISIC image ID; bổ sung metadata và audit patient-level/grouped split cùng manifest tái lập.
- Ghi đầy đủ nguồn, phiên bản, checksum và giấy phép dữ liệu; metadata provenance này hiện còn thiếu.
- Baseline YOLO26n-seg chính thức và toàn bộ ablation nêu trên.
- Kết quả đa seed, theo lớp, theo kích thước và robustness artifact.
- Latency/FPS trên phần cứng khai báo.
- External validation hoặc nêu rõ đây là giới hạn nếu chưa có.

Chỉ sau khi hoàn thành các mục này mới chuyển giả thuyết thành tuyên bố định lượng trong abstract, bảng kết quả và kết luận.

---

## Quy ước cập nhật tài liệu dự án

Từ ngày 03/08/2026, mọi quyết định liên quan đến cách gọi phương pháp, hướng phân tích, thay đổi kiến trúc, thay đổi code, lệnh thực thi, kết quả kiểm tra và kết quả thực nghiệm của dự án phải được ghi lại trong file `tailieu.md`. Không sử dụng một claim trong abstract hoặc bài báo nếu claim đó chưa được mô tả tại đây và chưa có bằng chứng tương ứng.

## Brief chi tiết để viết abstract nộp tối nay

### 1. Hướng phân tích trung tâm

Bài báo nên được viết theo hướng:

> **Phân đoạn thực thể tổn thương da bằng YOLO26 có đặc trưng độ phân giải cao, kết hợp nhiều biểu diễn ảnh soi da và cơ chế chú ý trên các tầng nông.**

Câu hỏi nghiên cứu chính không phải là “YOLO26 có tốt không”, mà là:

> Việc giữ lại đặc trưng stride-4 bằng P2 head, tái hiệu chỉnh đặc trưng P2/P3 bằng CBAM, và huấn luyện đồng thời trên ảnh gốc cùng ảnh đã xử lý artifact có giúp cải thiện phân đoạn tổn thương da, đặc biệt ở tổn thương nhỏ hoặc có biên khó, hay không?

Ba khó khăn thực tế được dùng để dẫn dắt bài:

1. Ảnh soi da có thể chứa lông, quầng tối của ống kính, sai lệch ánh sáng và nhiệt độ màu.
2. Các tầng dự đoán P3–P5 có thể làm mất một phần thông tin không gian chi tiết sau nhiều lần downsampling.
3. Phân bố chẩn đoán của HAM10000 không cân bằng; NV chiếm đa số, trong khi nhiều nhóm còn lại có ít mẫu hơn.

Không mô tả bài toán là “phân đoạn tế bào ung thư”. Đối tượng được phân đoạn là **tổn thương da (skin lesion)**. Mô hình thực hiện **instance segmentation**, đồng thời nhãn polygon mang một trong bảy class chẩn đoán.

### 2. Dữ liệu đầu vào và dữ liệu được phân tích

Nguồn đầu vào của pipeline:

`data/dataset_yolo_fixed_labels/dataset_yolo`

Số lượng đã ghi nhận:

| Split | Ảnh nguồn | Nhãn polygon |
|---|---:|---:|
| Train | 8.008 | 8.008 |
| Validation | 998 | 998 |
| Test | 1.007 | 1.007 |

Hai bộ dữ liệu **sau xử lý** mới là đối tượng phân tích trực tiếp của dự án:

1. `data/dataset_yolo_640x640_multiview`
2. `data/dataset_yolo_aug_p2_cbam`

Số lượng quan sát hiện tại:

| Bộ dữ liệu | Train | Validation | Test |
|---|---:|---:|---:|
| Multi-view | 16.016 | 998 | 1.007 |
| Augmented P2-CBAM | 31.880 | 998 | 1.007 |

Con số 31.880 là kết quả đã quan sát của lần sinh dữ liệu hiện có. Augmentation hiện chưa cố định random seed nên không tuyên bố đây là con số chắc chắn tái tạo được ở mọi lần chạy.

### 3. Pipeline xử lý dữ liệu đầy đủ

Tên nên dùng trong bài:

> **Artifact-oriented fixed multi-view dermoscopic preprocessing**  
> Tiền xử lý ảnh soi da đa phiên bản cố định, định hướng xử lý artifact.

Từ “fixed” rất quan trọng: pipeline áp dụng cùng một chuỗi xử lý cho mọi ảnh train; nó không có bộ phân loại để phát hiện loại artifact rồi mới lựa chọn phép xử lý.

#### Giai đoạn 1 — Multi-view preprocessing

Script: `data_processing/01_preprocess.py`.

Với mỗi ảnh thuộc tập train, pipeline sinh hai phiên bản có cùng nguồn ảnh và cùng ý nghĩa nhãn:

**View v1 — original/letterboxed view**

- Giữ nguyên nội dung ảnh và các artifact hiện hữu.
- Resize bảo toàn tỉ lệ và letterbox về 640 × 640.
- Biến đổi lại tọa độ polygon theo scale và padding.
- Mục tiêu: mô hình vẫn được tiếp xúc với ảnh gần với dữ liệu quan sát thực tế, thay vì chỉ học ảnh đã “làm sạch”.

**Quy ước cách hiểu:** v1 là nhánh đại diện cho **dataset mặc định/gần nguyên bản** trong bộ processed. Nó không phải bản sao byte-for-byte của ảnh nguồn vì đã được resize và letterbox về 640 × 640, nhưng chưa trải qua ROI crop, hair removal hoặc Gray-World. v7 là nhánh đã được preprocessing vật lý từ cùng ảnh nguồn.

**View v7 — artifact-processed view**

Tên `v1` và `v7` là mã phiên bản kế thừa của pipeline, không phải số thứ tự của bảy phép xử lý.

Chuỗi v7 gồm:

1. **ROI/vignette-oriented crop:** chuyển ảnh sang grayscale, threshold cố định tại 15, tìm external contour lớn nhất và crop theo bounding box. Mục tiêu là giảm quầng tối hoặc vùng biên ngoài trường quan sát của dermatoscope. Không gọi bước này là phát hiện ROI thông minh.
2. **DullRazor-inspired hair removal:** dùng morphological black-hat với kernel chữ thập 17 × 17, threshold nhị phân 10 để tạo hair mask, sau đó Telea inpainting với bán kính 3. Đây là triển khai lấy cảm hứng từ DullRazor, không khẳng định tương đương hoàn toàn thuật toán DullRazor gốc.
3. **Gray-World color constancy:** tính trung bình ba kênh B/G/R, đưa từng kênh về mức trung bình xám chung và clip về miền 0–255. Mục tiêu là giảm biến thiên màu do nguồn sáng hoặc thiết bị chụp.
4. **Letterbox 640 × 640:** resize bảo toàn tỉ lệ, padding và biến đổi polygon tương ứng sau crop.

Quy tắc theo split:

- Train: sinh cả v1 và v7, từ 8.008 ảnh thành 16.016 ảnh.
- Validation: chỉ sinh v1 letterbox, không có v7.
- Test: chỉ sinh v1 letterbox, không có v7.

Thiết kế này giúp validation/test không nhận lợi ích trực tiếp từ chuỗi loại artifact. Tuy nhiên, đây mới là thiết kế bảo toàn split; vẫn cần audit metadata bệnh nhân để loại trừ patient-level leakage.

#### Giai đoạn 2 — NV-excluding image-level augmentation

Script: `data_processing/02_augment.py`.

Tên chính xác nên dùng:

> **NV-excluding image-level augmentation**  
> Tăng cường dữ liệu mức ảnh có loại trừ NV.

Không gọi đây là cân bằng lớp hoàn toàn vì code không đặt target frequency riêng cho từng class.

Quy tắc:

- Sao chép toàn bộ ảnh/nhãn từ bộ multi-view.
- Nếu một ảnh chứa class 5 (`nv`), không tạo augmentation cho ảnh đó.
- Nếu ảnh không chứa NV, thực hiện tối đa ba lần augment.
- Mọi ảnh không chứa NV được xử lý theo cùng một chính sách, không phân biệt cụ thể akiec, bcc, bkl, df, mel hay vasc.

Các phép biến đổi:

- Horizontal flip: xác suất 0,50.
- Vertical flip: xác suất 0,25.
- Affine thứ nhất: scale 0,85–1,15; translate ±5%; rotate ±30°; xác suất 0,90.
- Affine shear: ±10° theo x/y; xác suất 0,35.
- Random brightness/contrast: giới hạn 0,12; xác suất 0,50.
- Hue/Saturation/Value: 5/8/5; xác suất 0,25.
- Gaussian blur: kernel 3–5; xác suất 0,15.

Polygon segmentation được biến đổi đồng bộ bằng keypoints. Sau biến đổi:

- Polygon dưới ba điểm bị loại.
- Polygon có diện tích dưới 50 px² bị loại.
- Tọa độ được chuẩn hóa và clip về [0,1].
- Nếu một lần augment không còn polygon hợp lệ thì lần đó không được lưu và không được thử bù thêm.

Validation và test được sao chép nguyên trạng từ đầu ra Giai đoạn 1; không thực hiện augmentation.

### 4. Kiến trúc đề xuất

Tên mô hình:

> **SkinSeg-YOLO26-P2Attn**

Đây là YOLO26n instance segmentation thực, không phải YOLOv12 đổi tên. Các thành phần YOLO26 được giữ gồm `C3k2`, `C2PSA`, `SPPF`, chế độ `end2end`, `reg_max=1` và head `Segment26`.

#### Tác dụng của P2 head

YOLO26n-seg baseline dự đoán từ P3/P4/P5, tương ứng stride 8/16/32. Mô hình đề xuất bổ sung P2 với stride 4, tạo bốn mức dự đoán:

| Head | Stride | Feature map khi input 640 × 640 | Vai trò dự kiến |
|---|---:|---:|---|
| P2 | 4 | 160 × 160 | Giữ thông tin không gian chi tiết, hỗ trợ tổn thương nhỏ và biên mảnh |
| P3 | 8 | 80 × 80 | Đặc trưng kích thước nhỏ–trung bình |
| P4 | 16 | 40 × 40 | Đặc trưng trung bình–lớn |
| P5 | 32 | 20 × 20 | Ngữ nghĩa mức cao và vùng tổn thương lớn |

P2 head có ba tác dụng được kỳ vọng:

1. Giảm mức mất chi tiết do downsampling vì feature map P2 chỉ giảm kích thước bốn lần so với ảnh đầu vào.
2. Cung cấp vị trí dự đoán và mask coefficients ở stride 4 cho `Segment26`. Trong implementation hội nghị hiện tại, prototype mask không nhận trực tiếp P2 mà được tạo từ P3–P5 rồi upsample về stride 4; vì vậy không claim P2 trực tiếp sinh hoặc tinh chỉnh prototype.
3. Tạo điều kiện mô tả tốt hơn các tổn thương nhỏ, đường biên bất quy tắc hoặc chuyển tiếp màu yếu.

Không được viết rằng P2 “chắc chắn cải thiện” hoặc “phân đoạn tới từng pixel chính xác hơn” trước khi có ablation B0–B1. P2 cũng làm tăng số điểm dự đoán, bộ nhớ và chi phí tính toán; phải báo params, FLOPs, latency và VRAM thực tế.

YOLO26 chính thức đã có topology P2 cho object detection. Đóng góp của dự án không phải phát minh P2, mà là **thích nghi topology P2 thành head phân đoạn bốn mức P2–P5 và đánh giá nó trong bài toán tổn thương da**.

#### Tác dụng của CBAM

CBAM được đặt sau đặc trưng P2 và P3 trong backbone, tức các tầng còn giữ độ phân giải không gian tương đối cao.

- Channel attention dùng average pooling và max pooling để tái trọng số kênh đặc trưng.
- Spatial attention kết hợp bản đồ mean/max theo chiều kênh rồi dùng convolution 7 × 7 để tái trọng số vị trí không gian.
- CBAM giữ nguyên kích thước tensor và số kênh.

Vai trò dự kiến là làm nổi bật đặc trưng liên quan đến tổn thương và giảm ảnh hưởng của nền/artifact. Không diễn giải channel attention là trực tiếp phát hiện melanin hoặc hemoglobin nếu chưa có thí nghiệm giải thích đặc trưng.

#### Luồng tổng thể

```text
Ảnh HAM10000 + polygon
        │
        ├── v1: Letterbox 640 × 640
        │
        └── v7: Threshold/largest-contour crop
                 → DullRazor-inspired hair removal
                 → Gray-World normalization
                 → Letterbox 640 × 640
        │
        └── Train only: NV-excluding image-level augmentation
                         → polygon transformation/filtering
        │
        ▼
YOLO26n backbone: C3k2 → CBAM@P2/P3 → C2PSA
        │
        ▼
P2/P3/P4/P5 feature pyramid
        │
        ▼
Segment26 → class + bounding box + lesion mask
```

### 5. Đóng góp nên trình bày trong abstract

Chỉ nên trình bày ở mức “propose/design/evaluate”:

1. Đề xuất một framework kết hợp ảnh nguyên trạng và ảnh được xử lý artifact để học từ hai biểu diễn bổ trợ.
2. Thích nghi YOLO26n thành kiến trúc instance segmentation bốn mức P2–P5.
3. Tích hợp CBAM tại P2/P3 để tái hiệu chỉnh đặc trưng độ phân giải cao.
4. Áp dụng NV-excluding image-level augmentation và biến đổi polygon đồng bộ.

Không dùng các claim:

- “the first”; 
- “completely removes artifacts”; 
- “solves class imbalance”; 
- “outperforms state of the art”; 
- “real-time” nếu chưa đo latency/FPS;
- mAP, Dice, IoU hoặc phần trăm chưa có file kết quả.

### 6. Khung abstract để người viết triển khai

Abstract khoảng 200–250 từ nên có năm phần, thường viết thành một đoạn:

1. **Background — 2 câu:** nêu vai trò của phân đoạn tổn thương da và ba khó khăn: artifact, biên/tổn thương nhỏ, phân bố chẩn đoán lệch.
2. **Data pipeline — 2–3 câu:** nêu hai train views; v1 letterbox và v7 gồm crop, DullRazor-inspired hair removal, Gray-World, letterbox; nêu NV-excluding augmentation với polygon transformation. Không cần liệt kê mọi xác suất trong abstract.
3. **Architecture — 2–3 câu:** nêu kiến trúc dựa trên Ultralytics YOLO26n-seg, bổ sung P2 stride 4 vào P3–P5, CBAM ở P2/P3 và `Segment26` bốn mức.
4. **Evaluation — 1–2 câu:** nói framework “is evaluated/will be evaluated” trên các split đã giữ cố định bằng mask mAP, Dice/IoU, precision/recall, latency và ablation. Nếu đây là abstract proposal chưa có kết quả, dùng “will be evaluated” hoặc “we define an evaluation protocol”; không giả metric.
5. **Significance — 1 câu:** nói thiết kế hướng tới bảo toàn chi tiết và tăng khả năng chịu biến thiên artifact, nhưng dùng “aims to” hoặc “is designed to”, không dùng “demonstrates” khi chưa có số liệu.

### 7. Các thông tin người viết abstract phải giữ nguyên

- Tên nền tảng: **YOLO26n**, không phải YOLOv12.
- Nhiệm vụ: **skin lesion instance segmentation**.
- Dataset: HAM10000-derived YOLO segmentation dataset với bảy class.
- Input source: 8.008/998/1.007 ảnh train/val/test.
- Processed observations: 16.016 train multi-view và 31.880 train augmented; val/test 998/1.007.
- P2 là stride 4; bốn head có stride 4/8/16/32.
- Attention: CBAM tại P2 và P3.
- Val/test chỉ letterbox; không v7 và không augmentation.
- 31.880 là observed count, không phải guarantee.
- Chưa có metric thực nghiệm đủ để tuyên bố vượt baseline.

### 8. Việc cần làm ngay sau khi nộp abstract

1. Bổ sung metadata bệnh nhân và audit patient-level split.
2. Chốt seed và môi trường tái lập.
3. Chạy baseline YOLO26n-seg chính thức.
4. Chạy P2-only, CBAM-only hoặc thiết kế ablation tương ứng, multi-view-only, augmentation-only và full model.
5. Đánh giá trên test đúng một lần sau khi chốt mô hình bằng validation.
6. Thu thập mask mAP50, mask mAP50–95, precision, recall, Dice/IoU, kết quả theo class, params, FLOPs, VRAM, latency và FPS.
7. Lưu `results.csv`, `best.pt`, `args.yaml`, seed, phiên bản thư viện và bảng tổng hợp vào dự án; cập nhật toàn bộ kết quả vào `tailieu.md` trước khi viết claim.

---

## Khảo sát công trình liên quan và kiểm tra khoảng trống nghiên cứu

### Phạm vi tìm kiếm

Literature-gap check được thực hiện ngày 03/08/2026 theo các nhóm từ khóa:

- `skin lesion segmentation + HAM10000`;
- `skin lesion segmentation + CBAM`;
- `YOLO + skin lesion + segmentation`;
- `YOLO + P2 head + CBAM`;
- `DullRazor + Gray World + skin lesion`;
- `original and hair-removed images + skin lesion`;
- `YOLO26 + P2 + segmentation`.

Kết luận “chưa tìm thấy” dưới đây chỉ áp dụng cho phạm vi tìm kiếm có mục tiêu này; nó không chứng minh tuyệt đối rằng không tồn tại công trình tương tự trong mọi cơ sở dữ liệu hoặc ngôn ngữ.

### Những thành phần đã có tiền lệ rõ ràng

| Thành phần của dự án | Tiền lệ đã tìm thấy | Ý nghĩa đối với novelty |
|---|---|---|
| HAM10000 | Tschandl và cộng sự công bố HAM10000 gồm 10.015 ảnh soi da đa nguồn năm 2018. | Dataset không mới. Phải trích dẫn bài dataset gốc và giải thích nguồn polygon của phiên bản dự án. |
| DullRazor/hair removal | DullRazor được Lee và cộng sự công bố từ năm 1997. Nhiều nghiên cứu sau dùng morphological black-hat, threshold và inpainting cho ảnh da. | Không claim phát minh loại lông. Code dự án phải được gọi là `DullRazor-inspired` vì không tái hiện đầy đủ bản gốc. |
| Gray-World/color constancy | Color constancy, Gray-World và Shades-of-Gray đã được dùng để giảm sai khác nguồn sáng trong ảnh da. | Không phải contribution riêng; chỉ là một thành phần preprocessing. |
| CBAM | Woo và cộng sự công bố CBAM tại ECCV 2018. Nhiều mô hình U-Net/encoder–decoder đã dùng CBAM cho skin-lesion segmentation. | Không được claim CBAM cho tổn thương da là mới. Novelty chỉ có thể nằm ở vị trí tích hợp, kiến trúc tổng thể và bằng chứng ablation. |
| Multiscale/high-resolution feature fusion | U-Net 3+, pyramid, skip connections và nhiều mạng skin-lesion segmentation đã kết hợp đặc trưng đa mức để phục hồi biên. | Không được nói chung chung rằng multi-scale segmentation là mới. |
| P2 head | Ultralytics cung cấp topology YOLO26-P2 cho detection. Nhiều bài YOLO ở UAV, khuyết tật bề mặt và small-object detection đã kết hợp P2 với CBAM/attention. | Không claim phát minh P2 hoặc tổ hợp P2+CBAM nói chung. |
| Hair-removed versus original images | Đã có nghiên cứu so sánh mô hình trên ảnh gốc và ảnh loại lông; cũng có nghiên cứu tạo dataset hairless riêng. | Ý tưởng dùng cả ảnh gốc và ảnh xử lý không hoàn toàn mới. Cần nhấn vào huấn luyện đồng thời hai view cùng source và đánh giá kiểm soát trong kiến trúc YOLO26 segmentation. |
| HAM10000 segmentation | MFSNet, foundation models, SAM-based frameworks, CBAMSNet và MambaLiteUNet đã báo cáo segmentation trên HAM10000. | Không claim là công trình đầu tiên phân đoạn HAM10000. Cần kiểm tra tính tương đồng của mask/split trước khi so sánh số liệu. |

### Các công trình gần nhất cần đặt cạnh bài của mình

#### 1. MFSNet — Multi Focus Segmentation Network

- Bài toán: supervised skin-lesion segmentation.
- Dataset có HAM10000, PH2 và ISIC 2017.
- Liên quan: khai thác nhiều mức/multiple focus cho segmentation.
- Khác dự án: không phải YOLO26 P2–P5 instance segmentation và không dùng pipeline v1/v7 của dự án.
- Cách dùng trong bài: baseline hoặc related work cho multi-scale lesion segmentation.

Nguồn: [MFSNet, arXiv:2203.14341](https://arxiv.org/abs/2203.14341).

#### 2. CBAMSNet

- Bài toán: lightweight skin-lesion segmentation.
- Thành phần: dynamic convolution, CBAM-based multiscale attention và cross-scale attention bridging.
- Dataset: ISIC 2017, ISIC 2018 và HAM10000.
- Mức độ gần: rất gần về từ khóa “CBAM + multiscale + HAM10000 segmentation”.
- Khác dự án: CBAMSNet là kiến trúc segmentation riêng, không dựa trên YOLO26, không dùng `Segment26` bốn head P2–P5 và không mô tả cùng pipeline v1/v7.
- Hệ quả: abstract của mình không được viết “CBAM-based multiscale attention for HAM10000 is novel”.

Nguồn: [CBAMSNet, Biomedical Signal Processing and Control, DOI 10.1016/j.bspc.2025.109239](https://doi.org/10.1016/j.bspc.2025.109239).

#### 3. SkinAttn-Net và các mô hình U-Net + CBAM

- SkinAttn-Net kết hợp CBAM, SE và ViT bottleneck trong encoder–decoder.
- ASCU-Net, Attention DenseUNet, SegNet3+ và nhiều biến thể khác cũng dùng channel/spatial attention cho biên tổn thương.
- Hệ quả: CBAM trong skin lesion là related work trực tiếp, không phải novelty độc lập.

Nguồn tham khảo:

- [SkinAttn-Net, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12852886/).
- [ASCU-Net, Diagnostics 2021](https://www.mdpi.com/2075-4418/11/3/501).
- [SegNet3+, DOI 10.1049/ipr2.70389](https://doi.org/10.1049/ipr2.70389).
- [U-Net-based Models for Skin Lesion Segmentation: More Attention and Augmentation, arXiv:2210.16399](https://arxiv.org/abs/2210.16399).

#### 4. YOLO và attention trong phân tích tổn thương da

- CAM-YOLO dùng YOLOv8 + CBAM cho classification đa phương thức, không phải instance segmentation.
- SkinDet-YOLO dùng YOLOv8 với context-aware multiscale fusion và boundary-aware detection, có nhắc channel/spatial attention.
- Đã có công trình cũ kết hợp YOLO và GrabCut cho skin-lesion segmentation.
- Hệ quả: không claim “lần đầu dùng YOLO/attention cho tổn thương da”.

Nguồn tham khảo:

- [CAM-YOLO, DOI 10.1109/ICCVDM66874.2025.11290051](https://doi.org/10.1109/ICCVDM66874.2025.11290051).
- [SkinDet-YOLO, Frontiers in Oncology](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2026.1840189/full).

#### 5. P2 + CBAM trong YOLO ngoài lĩnh vực da liễu

- Một mô hình YOLOv11 cho small-UAV detection đã kết hợp residual CBAM và P2 head để giữ feature stride 4.
- Các mô hình khuyết tật bề mặt và small-object detection cũng đã ablate P2 high-resolution head cùng CBAM/attention.
- Mức độ gần về kiến trúc: cao; khác biệt chủ yếu nằm ở domain và detection so với instance segmentation.
- Hệ quả: tổ hợp “P2 + CBAM” không đủ để làm novelty claim. Bài phải nhấn vào **YOLO26 four-scale instance segmentation adaptation, vị trí CBAM P2/P3, pipeline dữ liệu và đánh giá trên tổn thương da**.

Nguồn tham khảo:

- [YOLOv11 Residual Attention + P2 for small UAVs, Journal of Imaging](https://doi.org/10.3390/jimaging12030140).
- [YOLOv11-LCCAP2, DOI 10.1007/s11554-026-01885-1](https://doi.org/10.1007/s11554-026-01885-1).

#### 6. Preprocessing hair/artifact

- DullRazor là phương pháp kinh điển từ năm 1997.
- SkinNet-16 đã dùng grayscale, morphological black-hat và inpainting trong pipeline HAM10000 classification.
- Nghiên cứu hair-removal gần đây đã so sánh ảnh gốc với ảnh hairless hoặc dùng mô hình chuyên biệt để loại artifact.
- Một tổng quan chỉ ra DullRazor và morphological black-hat đều có thể làm mất một phần đặc trưng tổn thương; do đó giữ đồng thời v1 và v7 là một giả thuyết hợp lý cần kiểm chứng, không phải một kết luận có sẵn.

Nguồn tham khảo:

- [DullRazor, Computers in Biology and Medicine, DOI 10.1016/S0010-4825(97)00020-6](https://doi.org/10.1016/S0010-4825(97)00020-6).
- [SkinNet-16, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9395205/).
- [Automatic Skin Cancer Detection review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10672549/).
- [Hair and artifact removal using deep learning, Scientific Reports 2026](https://www.nature.com/articles/s41598-026-44545-x).

#### 7. Robustness và subgroup evaluation trên HAM10000

- Một công trình ICCV Workshop 2025 đánh giá foundation segmentation models trên toàn bộ HAM10000 theo diagnosis, tuổi, giới tính, vị trí giải phẫu và các thuộc tính thị giác như hair/dark corner.
- Đây là related work quan trọng cho hướng robustness. Bài của mình nên đánh giá theo subgroup artifact hoặc lesion size thay vì chỉ báo cáo overall mAP/Dice.

Nguồn: [Evaluating the Trustworthiness of Foundation Models for Skin Lesion Segmentation, ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/BISCUIT/html/Chu_Evaluating_the_Trustworthiness_of_Foundation_Models_for_Skin_Lesion_Segmentation_ICCVW_2025_paper.html).

#### 8. YOLO26

- YOLO26 là họ mô hình mới có detection, instance segmentation và các task-specific heads.
- Ultralytics cung cấp `yolo26-p2.yaml` cho object detection, nhưng P2 weights riêng không được phát hành; segmentation chuẩn vẫn dùng cấu hình riêng.
- Trong tìm kiếm mục tiêu hiện tại, chưa tìm thấy bài peer-reviewed áp dụng đúng **YOLO26n + custom P2/P3/P4/P5 Segment26 + CBAM P2/P3 + v1/v7 training** cho HAM10000 lesion instance segmentation.
- Vì YOLO26 rất mới, câu “chưa tìm thấy” phải được cập nhật lại ngay trước ngày nộp full paper.

Nguồn:

- [Ultralytics YOLO26 official documentation](https://docs.ultralytics.com/models/yolo26).
- [Ultralytics YOLO26 paper, arXiv:2606.03748](https://arxiv.org/abs/2606.03748).

### Kết luận novelty sau khảo sát

Không nên claim mới ở cấp từng thành phần:

- không mới: HAM10000;
- không mới: hair removal/DullRazor-inspired processing;
- không mới: Gray-World;
- không mới: CBAM trong skin-lesion segmentation;
- không mới: multiscale segmentation;
- không mới: P2 cho small-object detection;
- không mới: tổ hợp P2 + CBAM nói chung;
- không mới: so sánh ảnh gốc và ảnh đã loại lông.

Khoảng trống có thể bảo vệ, với điều kiện có thí nghiệm đầy đủ:

1. **Architecture adaptation:** thích nghi kiến trúc instance segmentation dựa trên Ultralytics YOLO26n thành `Segment26` bốn mức P2/P3/P4/P5 cho tổn thương da.
2. **High-resolution attention placement:** đặt CBAM cụ thể tại P2/P3 và kiểm chứng bằng ablation dưới cùng protocol.
3. **Paired multi-view learning:** huấn luyện trên cả v1 giữ artifact và v7 xử lý artifact của cùng source image, thay vì chỉ thay toàn bộ dataset bằng ảnh hairless.
4. **Joint controlled evaluation:** đánh giá riêng và kết hợp P2, CBAM, v7 và NV-excluding augmentation, đồng thời báo chi phí tính toán.
5. **Artifact/size-stratified analysis:** kiểm tra hiệu ứng theo lesion size và subgroup artifact, không chỉ metric tổng.

Câu novelty an toàn cho abstract:

> We propose an integrated YOLO26-based lesion instance-segmentation framework that combines a stride-4 P2 prediction path, early-stage CBAM refinement, and paired original/artifact-processed training views, and define a controlled ablation protocol to evaluate their individual and joint effects.

Câu này chỉ nói “đề xuất một framework tích hợp”; không nói từng module là mới hoặc đây là công trình đầu tiên.

### Baseline cần bổ sung sau literature check

Ngoài ablation nội bộ B0–B4, nên có ít nhất các nhóm baseline:

1. Official YOLO26n-seg.
2. Một baseline U-Net nhẹ hoặc U-Net/ResNet.
3. Một attention segmentation baseline có CBAM nếu có thể tái lập.
4. Một mô hình lightweight gần đây như MFSNet/CBAMSNet hoặc kết quả trích dẫn có cùng split/mask; nếu split khác thì chỉ đặt trong related work, không so sánh trực tiếp.
5. Nếu đủ tài nguyên, một foundation model như SAM/MedSAM để đặt bối cảnh hiện đại.

Tuyệt đối không đặt số từ bài khác cạnh số của mình như so sánh trực tiếp nếu khác mask provenance, split, patient grouping, resolution hoặc metric aggregation.

---

## Thiết kế nâng cấp được chọn: Paired Artifact-View Consistency + P2 Boundary Supervision

### Quyết định thiết kế

Ngày 03/08/2026, dự án chốt hướng nâng cấp tiếp theo:

1. **Đóng góp chính:** học nhất quán giữa cặp v1–v7 của cùng một ảnh nguồn ở mức biểu diễn tổn thương.
2. **Đóng góp phụ nhẹ:** thêm giám sát đường biên từ P2 trong quá trình train.
3. Chưa triển khai pixel-wise mask consistency giữa v1 và v7 cho tới khi pipeline lưu đầy đủ metadata crop/scale/padding để inverse-warp chính xác.

Tên làm việc:

> **SkinSeg-YOLO26-P2Attn-AVC: Paired Artifact-View Consistency and Boundary-Supervised High-Resolution Skin Lesion Instance Segmentation**

Tên này là tên thiết kế nội bộ. Chỉ dùng “artifact-invariant” hoặc “artifact-robust” trong tiêu đề chính thức sau khi thí nghiệm xác nhận.

### Xác nhận dữ liệu paired hiện có

Kiểm tra tên file thực tế cho thấy `dataset_yolo_640x640_multiview/images/train` chứa:

- 8.008 file v1;
- 8.008 file v7;
- 8.008 ISIC ID có đủ cả hai view;
- không có ID chỉ có v1 hoặc chỉ có v7.

Ví dụ:

```text
ISIC_0024307_v1.jpg  ← dataset mặc định/gần nguyên bản, letterbox
ISIC_0024307_v7.jpg  ← cùng ảnh nguồn, đã preprocessing
```

Trong `dataset_yolo_aug_p2_cbam`, tổng số file train là 31.880. Các file vẫn truy ngược được về 8.008 cặp nguồn, nhưng các bản `_aug0/_aug1/_aug2` của v1 và v7 đã trải qua phép biến đổi ngẫu nhiên độc lập. Vì vậy giai đoạn consistency đầu tiên chỉ sử dụng **8.008 cặp base v1–v7 không có hậu tố `_augk`**. Các ảnh augmented còn lại tiếp tục tham gia supervised YOLO training thông thường.

### Vì sao không so trực tiếp mask v1 và v7 theo pixel

v1 và v7 không cùng hệ tọa độ pixel:

- v1 chỉ letterbox ảnh nguồn;
- v7 crop theo contour trước khi letterbox;
- scale, crop origin và padding của hai view khác nhau.

Do đó phép so sánh trực tiếp `M_v1(x,y)` với `M_v7(x,y)` là sai hình học. Thiết kế giai đoạn đầu so sánh **lesion-level representations** sau ROI pooling độc lập theo ground-truth polygon của mỗi view. Pixel-mask consistency chỉ là nâng cấp giai đoạn sau khi lưu transformation metadata.

### Kiến trúc Paired Artifact-View Consistency (AVC)

Hai view đi qua cùng một mô hình và dùng chung toàn bộ trọng số:

```text
v1 ──┐
     ├── Shared SkinSeg-YOLO26-P2Attn ── prediction v1
v7 ──┘                                  └─ prediction v7

P2/P3 feature v1 + ROI polygon v1 → ROIAlign/Masked Pooling → z_v1
P2/P3 feature v7 + ROI polygon v7 → ROIAlign/Masked Pooling → z_v7

z_v1 ↔ z_v7 → Artifact-View Consistency Loss
```

Quy trình chi tiết:

1. Tạo `pair_id` bằng ISIC ID sau khi bỏ `_v1` hoặc `_v7`.
2. `PairedViewSampler` đưa v1 và v7 của cùng `pair_id` vào cùng paired batch.
3. Mỗi view được forward độc lập qua shared YOLO26n backbone.
4. Lấy feature tại P2 và P3 vì đây là nơi có CBAM và còn giữ chi tiết không gian cao.
5. Dùng polygon ground truth của từng view để tạo lesion ROI trong đúng hệ tọa độ của view đó.
6. Thực hiện ROIAlign hoặc masked global average pooling riêng trên P2/P3.
7. Ghép feature P2/P3 rồi đưa qua projection head nhỏ: Linear/Conv → normalization → vector `z`.
8. Tối thiểu hóa khoảng cách cosine giữa `z_v1` và `z_v7`.

Loss consistency cơ bản:

\[
\mathcal{L}_{AVC}=1-\frac{z_{v1}\cdot z_{v7}}{\lVert z_{v1}\rVert_2\lVert z_{v7}\rVert_2+\epsilon}
\]

Projection head chỉ dùng khi train và có thể bỏ khi inference, vì vậy không làm tăng latency của mô hình triển khai.

Để tránh representation collapse, `L_AVC` không được train độc lập; nó luôn đi cùng supervised detection/segmentation/classification losses. Có thể thử stop-gradient ở một nhánh hoặc thêm variance regularization nếu ablation cho thấy collapse.

### P2 Boundary Supervision nhẹ

Hướng 2 được sử dụng ở mức nhẹ: không tạo một decoder biên lớn và không thay đổi inference output chính.

Thiết kế:

1. Lấy P2 feature sau CBAM.
2. Thêm một auxiliary boundary head nhỏ trong lúc train: `3×3 Conv → activation → 1×1 Conv`.
3. Rasterize polygon ground truth thành binary mask.
4. Tạo boundary target bằng morphological gradient hoặc chênh lệch giữa dilation và erosion với bán kính cố định.
5. Resize boundary target về kích thước P2 160 × 160.
6. Dùng kết hợp weighted BCE và soft Dice cho boundary map.
7. Có thể loại auxiliary head khi export/inference; mục tiêu của head là điều hướng feature P2 trong train.

Loss boundary:

\[
\mathcal{L}_{bd}=\mathcal{L}_{WBCE}(B,\hat{B})+\mathcal{L}_{Dice}(B,\hat{B})
\]

Loss tổng:

\[
\mathcal{L}_{total}=\mathcal{L}_{YOLO26}+\lambda_{AVC}\mathcal{L}_{AVC}+\lambda_{bd}\mathcal{L}_{bd}
\]

Giá trị khởi đầu để khảo sát, không phải hyperparameter đã xác nhận:

- `lambda_AVC ∈ {0.05, 0.10, 0.20}`;
- `lambda_bd ∈ {0.05, 0.10}`;
- warm-up auxiliary losses trong 5–10 epoch đầu để tránh phá pretrained representation.

### Data flow khi train

Nên dùng hai loại batch xen kẽ:

1. **Standard supervised batch:** lấy toàn bộ dữ liệu 31.880 ảnh; tối ưu `L_YOLO26` và `L_bd` nếu có boundary target.
2. **Paired consistency batch:** lấy 8.008 cặp base v1–v7; tối ưu supervised loss cho từng view cộng `L_AVC` và `L_bd`.

Không áp dụng consistency cho hai file chỉ vì chúng có cùng `_augk`, do augmentation hiện được lấy mẫu ngẫu nhiên độc lập và không lưu transformation metadata chung.

### Tác dụng dự kiến

AVC hướng tới:

- làm feature tổn thương ít nhạy hơn với lông, vignette và thay đổi màu do preprocessing;
- tránh việc mô hình coi v1 và v7 như hai mẫu hoàn toàn không liên quan;
- giữ thông tin chung về tổn thương trong khi supervised loss vẫn bảo toàn class và mask;
- không tăng chi phí inference nếu projection head chỉ tồn tại lúc train.

Boundary supervision hướng tới:

- tận dụng P2 stride 4 cho biên mảnh;
- giảm mask có biên răng cưa hoặc bị co/giãn quá mức;
- tạo lý do kiến trúc rõ ràng cho việc thêm P2, thay vì chỉ nói P2 hỗ trợ small objects.

Tất cả các tác dụng trên là giả thuyết cần thí nghiệm; không viết thành kết quả trong abstract hiện tại.

### Rủi ro kỹ thuật và biện pháp kiểm soát

| Rủi ro | Biện pháp |
|---|---|
| v1/v7 khác hình học | Consistency ở lesion representation sau ROI pooling; không so pixel trực tiếp. |
| Preprocessing v7 làm mất đặc trưng có ích | Supervised loss áp dụng cho cả v1/v7; ablate v1-only, v7-only và paired. |
| Consistency làm feature collapse | Giữ supervised loss, projection normalization, kiểm tra variance; thử stop-gradient. |
| Boundary pixels quá ít | Weighted BCE + Dice và boundary band dày cố định. |
| Auxiliary loss áp đảo main loss | Lambda nhỏ, warm-up và theo dõi riêng từng loss. |
| Batch paired tăng bộ nhớ | Server 24 GB; bắt đầu paired batch 8 hoặc 16 cặp, AMP, gradient accumulation nếu cần. |
| Patient-level leakage | Phải bổ sung metadata/group split trước khi dùng kết quả để claim. |

### Ablation bắt buộc cho phiên bản nâng cấp

| ID | P2 | CBAM | v1+v7 | AVC | Boundary supervision | Mục đích |
|---|---:|---:|---:|---:|---:|---|
| N0 | Không | Không | v1 | Không | Không | Official YOLO26n-seg baseline |
| N1 | Có | Không | v1 | Không | Không | Hiệu ứng P2 |
| N2 | Có | Có | v1 | Không | Không | Hiệu ứng CBAM |
| N3 | Có | Có | v1+v7 độc lập | Không | Không | Hiệu ứng multi-view data đơn thuần |
| N4 | Có | Có | v1+v7 paired | Có | Không | Đóng góp chính AVC |
| N5 | Có | Có | v1+v7 độc lập | Không | Có | Hiệu ứng boundary supervision |
| Full | Có | Có | v1+v7 paired | Có | Có | Hệ thống đầy đủ |

Thêm hai đối chứng dữ liệu nếu đủ tài nguyên:

- v1-only;
- v7-only.

### Metric bổ sung

Ngoài mask mAP và Dice/IoU, phiên bản này phải báo:

- Boundary IoU hoặc Boundary F-score;
- HD95 và ASSD nếu có thể triển khai đúng;
- cosine similarity giữa paired v1/v7 embeddings trước và sau AVC;
- chênh lệch metric giữa v1 và v7 của cùng source;
- kết quả theo lesion-size bin và artifact subgroup;
- chi phí train và chi phí inference sau khi bỏ auxiliary heads.

## Kiểm tra tính mới của thiết kế AVC + boundary

### Tiền lệ đã tồn tại

1. **Transformation-consistent skin-lesion segmentation:** đã có self-ensembling ép prediction nhất quán dưới rotation/flipping cho semi-supervised lesion segmentation. Nguồn: [Li et al., arXiv:1808.03887](https://arxiv.org/abs/1808.03887).
2. **Geometry-aware consistency:** đã có dual-view consistency kết hợp geometric constraints trong semi-supervised medical segmentation. Nguồn: [Liu and Zhao, arXiv:2202.06104](https://arxiv.org/abs/2202.06104).
3. **Dual-view medical segmentation:** Duo-SegNet dùng adversarial dual-view learning; một công trình wound segmentation dùng raw/enhanced dual-view semantic fusion với illumination correction học được. Nguồn: [Duo-SegNet, arXiv:2108.11154](https://arxiv.org/abs/2108.11154) và [Wound Segmentation with Dynamic Illumination Correction and Dual-view Semantic Fusion, arXiv:2207.05388](https://arxiv.org/abs/2207.05388).
4. **Boundary-aware skin-lesion segmentation:** BLA-Net, BDFormer, BAF-UNet và các mô hình mới đã dùng auxiliary boundary learning, boundary loss hoặc shape prior. Nguồn: [BLA-Net](https://doi.org/10.1016/j.cmpb.2022.107190), [BDFormer](https://doi.org/10.1016/j.artmed.2025.103079), [BAF-UNet](https://doi.org/10.1117/1.JMI.13.1.014003).
5. **Boundary/shape prior trên HAM10000:** đã có dual-prior network kết hợp boundary constraint và shape prior trên HAM10000. Nguồn: [SegMan-based dual-prior network, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12998886/).

### Phần không được claim mới

- consistency learning nói chung;
- dual-view medical segmentation nói chung;
- boundary loss hoặc boundary auxiliary head;
- CBAM, P2 hoặc YOLO segmentation;
- dùng raw/enhanced images nói chung.

### Khoảng trống cụ thể còn có thể bảo vệ

Trong phạm vi tìm kiếm mục tiêu tới ngày 03/08/2026, chưa tìm thấy công trình trùng hoàn toàn:

> **An Ultralytics YOLO26n-based four-scale P2–P5 instance-segmentation framework with CBAM at P2/P3, trained using paired default-letterboxed and deterministically artifact-processed views of the same HAM10000 image, with lesion-level artifact-view consistency and lightweight P2 boundary supervision.**

Điểm phân biệt quan trọng:

- Hai view không phải hai augmentation hình học ngẫu nhiên của cùng ảnh; chúng biểu diễn **artifact-preserved versus artifact-processed**.
- Consistency được áp dụng trên paired lesion embeddings sau ROI pooling trong đúng hệ tọa độ của từng view.
- Boundary supervision gắn trực tiếp với lý do sử dụng P2 stride 4.
- Projection/boundary auxiliary heads có thể bỏ khi inference.
- Toàn bộ hiệu ứng được tách bằng ablation N0–Full.

Đây vẫn là claim về **cấu hình tích hợp và protocol**, không phải claim “first”. Cần cập nhật systematic search trước khi nộp full paper.

### Novelty statement đưa cho người viết

Phiên bản an toàn:

> We extend a four-scale YOLO26 lesion instance-segmentation model with paired artifact-view consistency learning. Unlike treating the default and artifact-processed images as independent training samples, the proposed strategy explicitly aligns lesion-level representations extracted from paired views of the same source image, while lightweight P2 boundary supervision encourages high-resolution contour encoding. A controlled ablation protocol is designed to separate the effects of the P2 path, CBAM refinement, paired-view consistency, and boundary supervision.

Không đưa câu này vào abstract như một kết quả đã chứng minh. Nếu chưa triển khai AVC/boundary trước thời điểm nộp abstract, dùng tương lai hoặc proposal wording: `we propose`, `is designed to`, `will be evaluated`.

---

## Hiệu chỉnh thiết kế sau phản biện kỹ thuật (03/08/2026)

Phần này **thay thế các chi tiết triển khai còn mơ hồ ở thiết kế AVC phía trên**. Hướng nghiên cứu không đổi: đóng góp chính là paired artifact-view consistency; boundary supervision là đóng góp phụ nhẹ.

### 1. Các vấn đề đã phát hiện và quyết định sửa

1. Cosine loss dương giữa hai projection có thể hội tụ về vector hằng. Supervised YOLO loss không bảo đảm projection head không collapse. Vì vậy không dùng cosine-positive-only làm loss chính.
2. Ultralytics trainer mặc định không cung cấp đồng thời `pair_id`, feature trung gian P2/P3 và auxiliary loss. Việc triển khai phải có dataset/sampler, model hook, criterion và trainer tùy biến.
3. “P2/P3 sau CBAM” trước đây nhập nhằng giữa backbone và neck. Thiết kế chính thức dùng **neck prediction features**, không dùng trực tiếp backbone CBAM outputs.
4. Mosaic, MixUp và Copy-Paste làm mất quan hệ hình học/ownership giữa hai view. Chúng phải tắt trong paired batch hoặc được đồng bộ hóa có metadata; phiên bản đầu sẽ tắt trong paired batch.
5. Boundary target, sampling budget và evaluation v7 trước đây chưa đủ đặc tả. Các mục dưới đây khóa các định nghĩa này.
6. Pipeline crop hiện cần được audit về polygon clipping thật sự; clip từng đỉnh theo trục chưa tương đương phép giao polygon với crop rectangle.

### 2. Feature contract chính thức

Theo `models/yolo26n-seg-p2-cbam.yaml`, các tensor dùng cho auxiliary learning là:

- **Neck P2:** node 21, stride 4, output 32 channels ở scale `n`, kích thước `H/4 × W/4` (160×160 khi input 640×640).
- **Neck P3:** node 18, stride 8, output 64 channels ở scale `n`, kích thước `H/8 × W/8` (80×80 khi input 640×640).
- **Boundary head:** chỉ nhận neck P2 node 21.
- **AVC projector:** nhận vector ghép từ polygon-masked pooling của neck P2 và neck P3.

CBAM vẫn nằm tại backbone node 3 (P2, 64 channels) và node 6 (P3, 128 channels), tác động gián tiếp tới neck. Không gọi node 3/6 là tensor AVC để tránh nhập nhằng.

### 3. Paired Artifact-View Consistency có chống collapse

Với từng `pair_id`, polygon của v1 và v7 được rasterize độc lập trong đúng hệ tọa độ của view tương ứng. Mask được downsample bằng area interpolation tới kích thước feature; masked global average pooling được dùng thay ROIAlign để biểu diễn đúng hình dạng tổn thương. Vector P2 và P3 được nối rồi đưa qua cùng một MLP projector, tạo `z_v1, z_v7 ∈ R^D`.

Loss AVC chính thức dùng cấu trúc VICReg-style:

\[
L_{AVC}=\alpha L_{inv}+\beta L_{var}+\gamma L_{cov}
\]

- `L_inv`: mean squared error giữa `z_v1` và `z_v7`, lấy trung bình theo lesion rồi theo pair.
- `L_var`: phạt mỗi chiều embedding có standard deviation dưới 1 trong từng view: `mean(ReLU(1 - sqrt(var(z)+eps)))`.
- `L_cov`: bình phương các phần tử ngoài đường chéo của covariance matrix, chuẩn hóa theo số chiều embedding.
- Batch AVC phải có tối thiểu 16 pair thực; nếu VRAM không đủ, tính variance/covariance trên embedding đã gather qua các micro-batch trong một optimizer step.
- Giá trị khảo sát ban đầu sau khi chuẩn hóa từng thành phần: `alpha=1`, `beta=1`, `gamma=0.04`; đây là điểm khởi tạo, không phải kết quả tối ưu.

Nguồn cơ sở cho cơ chế chống collapse: [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906). Điểm mới của bài không nằm ở VICReg mà ở cách định nghĩa paired artifact-view lesion representation và kiểm chứng có kiểm soát trong pipeline này.

Nếu một ảnh có nhiều instance, AVC chỉ ghép instance khi hai label có cùng class và phép đối sánh polygon theo source-coordinate là duy nhất. Trước giai đoạn train phải audit số instance/class của toàn bộ 8.008 pair; pair không đối sánh chắc chắn sẽ bị loại khỏi AVC nhưng vẫn được dùng cho supervised loss.

### 4. Boundary target và loss chính thức

- Tạo một **class-agnostic union lesion mask** cho mỗi ảnh trong hệ tọa độ input sau letterbox.
- Boundary band là morphological gradient `dilate(mask,r) - erode(mask,r)` với `r=3 px` tại input 640; nếu input động, scale `r` theo cạnh ngắn và làm tròn tối thiểu 1 px.
- Downsample boundary về `H/4 × W/4` bằng max-pooling/adaptive max operation để không làm mất biên mảnh.
- Boundary head: `Conv 3×3 → SiLU → Conv 1×1`, output **logits** một kênh.
- `pos_weight = clamp(N_negative / max(N_positive,1), 1, 20)` tính theo batch.
- `L_bd = WBCEWithLogits + SoftDice(sigmoid(logits), target)`, Dice smoothing bằng 1, giảm trung bình theo từng ảnh rồi theo batch.

Boundary head là training-only và được bỏ khỏi graph export. Boundary supervision là kỹ thuật đã có tiền lệ, nên chỉ được trình bày là auxiliary design hỗ trợ P2, không claim phát minh mới.

### 5. Thành phần code bắt buộc

Triển khai sau này phải có các contract sau, thay vì chỉ gọi `model.train()` mặc định:

1. `PairedSegDataset` lập index 8.008 base pair và trả `pair_id`, hai ảnh, hai bộ polygon/class cùng metadata hình học.
2. `PairedBatchSampler` và custom collate giữ hai view của cùng source trong một batch.
3. Model wrapper/hook trả prediction chính cùng neck node 18/21 khi `training=True`; đường predict/export chuẩn không đổi.
4. Custom criterion bọc YOLO26 segmentation loss chính và cộng `L_AVC`, `L_bd`.
5. Custom segmentation trainer xây dataset/dataloader/model/criterion và log riêng từng loss.
6. Checkpoint train lưu projector và boundary head để resume; checkpoint deploy loại hai auxiliary head và được kiểm tra equivalence của prediction chính trước/sau khi strip.

Trong paired branch phiên bản đầu: `mosaic=0`, `mixup=0`, `copy_paste=0`; không dùng random crop/affine độc lập. Có thể dùng photometric jitter nhẹ độc lập vì không đổi polygon. Nếu thêm geometric augmentation thì cùng một transform phải áp lên ảnh và polygon của từng view, đồng thời lưu transform metadata.

### 6. Sampling và đối chứng công bằng

Định nghĩa một epoch theo **số optimizer updates cố định**, không theo một vòng duyệt dataset. Lịch khởi đầu dùng 1 standard update xen 1 paired update. Mọi thí nghiệm so sánh N3, paired-control, N4 và Full phải có:

- cùng số optimizer updates;
- cùng batch schedule và số image presentations;
- cùng seed/split;
- supervised loss của paired batch là trung bình của loss v1 và v7, không phải cộng đôi;
- cùng augmentation settings.

Đối chứng bắt buộc để đo riêng AVC là **paired batching nhưng `lambda_AVC=0`**. Như vậy N4 chỉ khác đối chứng ở loss AVC, không khác do thấy thêm ảnh hoặc do sampling. Đối chứng v7-only cũng bắt buộc nếu đủ thời gian. Báo trung bình, độ lệch chuẩn hoặc confidence interval qua ít nhất 3 seed cho mô hình chính và đối chứng trực tiếp.

### 7. Geometry audit trước train

Trước khi bật AVC phải sửa/kiểm tra phép crop label như sau:

- dùng true polygon–rectangle intersection (ví dụ Sutherland–Hodgman hoặc thư viện hình học tương đương), không chỉ clamp từng vertex;
- lưu cho mỗi view: source size, crop box, scale, pad và polygon trước/sau biến đổi;
- assert polygon hợp lệ, diện tích dương, nằm trong canvas và class/instance tương ứng giữa v1–v7;
- kiểm tra forward/inverse transform trên mẫu và báo round-trip error;
- xuất báo cáo số pair hợp lệ, bị loại và lý do.

Đây là publication blocker: không dùng paired-consistency result cho paper nếu geometry audit chưa đạt.

### 8. Evaluation v1/v7 và artifact subgroup

Benchmark chính vẫn chạy trên **val/test v1 chưa xử lý artifact** để giữ protocol thực tế. Đồng thời tạo `v7_eval` deterministically từ chính val/test, lưu ở thư mục riêng, tuyệt đối không đưa vào train. `v7_eval` chỉ phục vụ robustness analysis theo cặp:

- chênh lệch Dice/IoU/mAP giữa v1 và v7 cùng source;
- prediction consistency sau khi quy hai mask về source coordinates;
- Boundary F-score, HD95/ASSD nếu implementation được kiểm chứng;
- embedding distance chỉ dùng để phân tích, không dùng để tune trên test.

Artifact subgroup không được gắn nhãn bằng suy đoán cảm tính. Dùng các proxy tái lập từ pipeline: tỷ lệ hair-mask coverage, tỷ lệ diện tích bị crop do vignette và độ lớn color correction Gray-World; chia bin theo quartile được xác định trên train. Trong bài phải gọi đây là **algorithmic artifact proxies**, không phải clinical artifact annotations.

Phải kiểm tra metadata HAM10000 và cố định split theo lesion/patient group trước thí nghiệm cuối. Nếu không khôi phục được patient/lesion grouping đáng tin cậy, phải công khai giới hạn này và không claim patient-independent generalization.

### 9. Ablation hiệu chỉnh

| ID | Mô tả | So sánh dùng để kết luận |
|---|---|---|
| A0 | YOLO26n-seg chuẩn, v1 | baseline |
| A1 | + P2, v1 | A1−A0: P2 |
| A2 | + P2 + CBAM, v1 | A2−A1: CBAM |
| A3 | P2+CBAM, standard v1+v7 | data multi-view độc lập |
| A4-control | P2+CBAM, paired schedule, AVC=0 | kiểm soát sampler/exposure |
| A4 | như A4-control + VICReg-style AVC | A4−A4-control: đóng góp AVC |
| A5 | như A4-control + boundary | A5−A4-control: boundary |
| Full | AVC + boundary | Full−A4 và Full−A5: hiệu ứng kết hợp |
| D-v7 | P2+CBAM, v7-only | kiểm tra v7 riêng |

Các hàng so sánh trực tiếp phải dùng equal-update/equal-image budget. Không diễn giải A3−A2 chỉ là “lợi ích v7” nếu số lần nhìn ảnh khác nhau.

### 10. Kết luận tính mới sau khi xem lại

Các thành phần riêng lẻ **đều đã có tiền lệ**: P2/multiscale, CBAM, DullRazor/Gray-World, raw–enhanced dual-view learning, consistency regularization, VICReg và boundary-aware segmentation. Vì vậy không claim “first”, không claim boundary head là mới, và không dùng cụm “genuine YOLO26”.

Khoảng trống có thể bảo vệ là **formulation + controlled evidence**:

> An Ultralytics YOLO26n-based four-scale P2–P5 lesion instance-segmentation framework that learns anti-collapse lesion-level consistency from paired default-letterboxed and deterministically artifact-processed HAM10000 views, with lightweight P2 boundary supervision and exposure-controlled ablations.

Trong phạm vi tìm kiếm mục tiêu hiện tại, chưa tìm thấy bài trùng toàn bộ formulation/protocol này. Đây chỉ là kết quả tìm kiếm, không phải bằng chứng tuyệt đối rằng chưa từng có ai làm. Trọng tâm khi viết phải là: (i) pair được định nghĩa từ cùng source lesion nhưng khác trạng thái artifact; (ii) consistency ở lesion representation có chống collapse; (iii) boundary supervision liên kết có chủ đích với neck P2; và (iv) ablation kiểm soát sampler/exposure.

Novelty statement an toàn để chuyển cho người viết:

> We propose an Ultralytics YOLO26n-based four-scale lesion instance-segmentation framework that explicitly learns artifact-view-stable lesion representations from paired default-letterboxed and deterministically artifact-processed HAM10000 images. A collapse-resistant lesion-level consistency objective is coupled with lightweight supervision of the stride-4 neck feature. Exposure-controlled ablations are designed to distinguish the effects of multiview data, paired sampling, consistency learning, and boundary supervision. The contribution is the task-specific formulation and controlled evaluation protocol rather than any individual module.

Nếu abstract được nộp trước khi có kết quả, dùng `we propose`, `is designed to`, `will be evaluated`; không dùng `improves`, `outperforms`, `robust` hoặc số metric chưa chạy.

### 11. Kết quả phản biện thiết kế

Phản biện kỹ thuật lần hai kết luận thiết kế đã **ready for implementation planning**: bảy blocker ban đầu đã được xử lý, pixel-mask consistency vẫn được loại khỏi training và chỉ còn các chi tiết triển khai không chặn thiết kế (empty-mask handling, thống kê VICReg khi gradient accumulation, batch size chính xác, trạng thái bắt buộc của v7-only control và outer loss-weight schedule).

---

## Nhật ký đồng bộ GitHub (03/08/2026)

Yêu cầu: cập nhật mã nguồn và tài liệu lên `https://github.com/Hainguyen752004/SPKT_PAPER` để cộng tác viên viết bài.

Kiểm tra trước khi đồng bộ:

- `D:\PAPER_SPKT\Ham1000_p2_CBAM` chưa phải Git repository.
- Máy hiện không cài GitHub CLI (`gh`), nên dùng Git qua HTTPS.
- Remote tồn tại, branch mặc định là `main`, commit ban đầu `b4a500f`.
- Clone làm việc tại `D:\PAPER_SPKT\SPKT_PAPER_sync` để không thay đổi cấu trúc thư mục dữ liệu gốc.
- Chỉ đưa source code, model YAML, tests, tài liệu và báo cáo audit nhỏ lên GitHub.
- Không đưa dataset ảnh/label, thư mục `runs`, cache Python/Pytest hoặc weight `*.pt` lên GitHub vì kích thước lớn và để tránh nhân bản artifact huấn luyện.

Các lệnh kiểm tra chính:

```powershell
git ls-remote --symref https://github.com/Hainguyen752004/SPKT_PAPER.git HEAD
git clone https://github.com/Hainguyen752004/SPKT_PAPER.git D:\PAPER_SPKT\SPKT_PAPER_sync
```

Kết quả: remote truy cập đọc thành công và clone branch `main` thành công. Trạng thái commit/push cuối cùng được ghi bổ sung sau khi hoàn tất.

Kiểm tra bản đóng gói lần 1:

- Secret-pattern scan: không phát hiện token/API key/password rõ ràng.
- `python -m pytest -q`: 46 passed, 1 failed.
- Lỗi duy nhất: `test_packaged_pretrained_path_exists` yêu cầu `models/yolo26n-seg.pt`, trong khi quy tắc loại mọi file `*.pt` đã bỏ nhầm pretrained weight cần cho reproducibility.
- Cách xử lý: cho phép riêng `models/yolo26n-seg.pt` (khoảng 6,7 MB) trong `.gitignore`; vẫn loại các checkpoints/runs khác, sau đó chạy lại toàn bộ test.

Kiểm tra bản đóng gói lần 2 sau khi bổ sung pretrained weight:

- `python -m pytest -q`: **47 passed**, không có test failure; có 14 deprecation warnings từ dependency Matplotlib/PyParsing, không ảnh hưởng kết quả.

Kết quả publish GitHub:

- Commit nội dung chính: `b8ec656` — `Add YOLO26 P2-CBAM skin lesion paper workspace`.
- Push thành công từ `b4a500f` lên `b8ec656` trên `origin/main`.
- Repo cộng tác: `https://github.com/Hainguyen752004/SPKT_PAPER`.
- Sau dòng nhật ký này có thêm một documentation-only commit để lưu chính kết quả push vào `tailieu.md` trên remote.

---

## Bắt đầu triển khai code AVC (03/08/2026)

Quyết định triển khai theo từng publication gate, không viết toàn bộ trainer một lần. Phase 1 là **paired geometry audit**, vì consistency loss sẽ không đáng tin nếu polygon v7 bị clamp sai sau Smart ROI Crop.

Implementation plan chi tiết được lưu tại:

`docs/superpowers/plans/2026-08-03-paired-geometry-audit.md`

Thứ tự phase đã khóa:

1. true polygon clipping + reversible transform metadata;
2. audit đủ 8.008 pair và tạo v7_eval tách biệt;
3. paired dataset/sampler;
4. masked P2/P3 pooling + AVC/VICReg;
5. P2 boundary supervision;
6. custom YOLO26 trainer, ablation và export.

Dataset hiện có không bị overwrite; corrected dataset sẽ dùng destination versioned mới cho tới khi audit đạt.

### Điều chỉnh mục tiêu xuất bản: Conference trước, Q1 sau

Ngày 03/08/2026, tác giả quyết định chia nghiên cứu thành hai mốc:

#### Mốc 1 — Bài hội nghị hiện tại

Mục tiêu là hoàn thành một bài gọn, có một thông điệp chính rõ ràng và số liệu trung thực:

- nền tảng Ultralytics YOLO26n-based instance segmentation;
- prediction bốn mức P2–P5 và CBAM tại high-resolution stages;
- pipeline v1/v7 và NV-excluding augmentation đã có;
- bổ sung **paired artifact-view consistency** ở mức tối thiểu khả thi nếu kịp triển khai/ablation;
- boundary supervision chỉ đưa vào khi code và ablation đủ ổn định, không bắt buộc phải nhồi vào abstract hội nghị;
- benchmark chính trên v1 val/test, báo mask mAP, Dice/IoU, boundary metric và chi phí mô hình;
- tuyệt đối không dùng claim Q1, SOTA hoặc robustness nếu chưa có kết quả tương ứng.

Mức thí nghiệm tối thiểu cho conference:

1. YOLO26n-seg baseline;
2. +P2;
3. +P2+CBAM;
4. +v1/v7 multiview;
5. paired-control và +AVC nếu phase AVC hoàn tất;
6. ít nhất một lần chạy hoàn chỉnh cho tất cả hàng, sau đó ưu tiên 3 seed cho baseline trực tiếp và mô hình đề xuất.

#### Mốc 2 — Extended Q1 journal

Sau conference mới mở rộng thành bài journal bằng các phần đòi hỏi bằng chứng mạnh hơn:

- patient/lesion-group split được xác minh;
- external dataset hoặc cross-dataset validation;
- đầy đủ VICReg-style AVC, P2 boundary supervision và interaction ablation;
- nhiều seed/confidence interval;
- calibration, failure analysis, artifact subgroup và statistical significance;
- so sánh thêm mô hình chuyên biệt/foundation model;
- phân tích latency, FLOPs, VRAM và deployment;
- nếu phù hợp, pixel-level consistency có inverse-warp chính xác.

Như vậy bài conference là nền tảng thực nghiệm, còn Q1 là bản mở rộng có protocol và validation mạnh hơn; không chỉ kéo dài nội dung conference bằng cách thêm vài bảng.

### Phản biện implementation plan phase 1

Reviewer chưa duyệt bản plan geometry đầu tiên vì sáu contract cần đặc tả rõ trước khi code:

1. metadata phải lưu stable instance ID, class và polygon trước/sau transform;
2. publication gate phải nêu điều kiện pass/fail và tolerance;
3. rasterization/downsample P2 mask phải có quy tắc thực thi chính xác;
4. CLI destination versioned phải được hỗ trợ và kiểm thử an toàn;
5. Git commits phải thực hiện trong repo sync `D:\PAPER_SPKT\SPKT_PAPER_sync`, vì thư mục dữ liệu gốc không có `.git`;
6. metadata phải định nghĩa công thức hair coverage, vignette crop ratio và Gray-World correction magnitude.

Các điểm này sẽ được sửa trong plan trước khi thay đổi source. Đây không làm đổi hướng conference; nó ngăn lỗi nhãn và làm số liệu hội nghị có thể kiểm chứng.

### Kết quả rà soát tài liệu và duyệt implementation plan

Ngày 03/08/2026, đã kiểm tra `tailieu.md` và plan geometry bằng bộ đọc UTF-8 nghiêm ngặt:

- cả hai file đọc thành công dưới UTF-8;
- không có Unicode replacement character `U+FFFD`;
- không phát hiện các lỗi gõ đã biết như `xữ`, `heah`, `prosec`, `nolp`, `khum`, `rùi`;
- thay các cụm cũ `genuine YOLO26` và `physically artifact-processed` bằng cách gọi trung tính, chính xác hơn: `Ultralytics YOLO26n-based` và `deterministically artifact-processed`;
- chuẩn hóa cụm `P2 stride-4` thành `P2 stride 4` trong phần hướng dẫn tiếng Việt.

Reviewer đã duyệt (`APPROVED`) implementation plan sau khi bổ sung đầy đủ sáu contract về metadata instance, publication gate, P2 rasterization, destination safety, Git workflow và artifact proxies. Plan hiện sẵn sàng để thực thi.

Kết quả đồng bộ GitHub:

- Regression trước commit: **47 passed**, 14 dependency deprecation warnings, không có failure.
- `git diff --check`: không có lỗi whitespace.
- Commit: `e707521` — `docs: define conference scope and geometry implementation plan`.
- Push thành công lên `origin/main` của `Hainguyen752004/SPKT_PAPER`.

### Quyết định workspace khi triển khai conference

Quy trình ban đầu đề xuất Git worktree để cô lập nhánh phát triển. Tác giả chọn triển khai trực tiếp tại `D:\PAPER_SPKT\Ham1000_p2_CBAM` để thuận tiện truy cập dataset cục bộ. Vì thư mục này không phải Git repository, quy tắc thực thi được điều chỉnh:

- chỉ sửa source/test/documentation bằng patch có kiểm soát;
- không overwrite hoặc xóa hai processed dataset hiện tại;
- chạy targeted tests và full regression tại thư mục gốc;
- chỉ sau khi test đạt mới đồng bộ các file đã review sang `D:\PAPER_SPKT\SPKT_PAPER_sync` để commit/push;
- nếu một task thất bại, giữ nguyên log lỗi trong `tailieu.md` và sửa theo TDD, không che giấu kết quả.

Bắt đầu Task 1 của conference implementation: true polygon–rectangle clipping và unit tests.

#### Kết quả Task 1 — True polygon clipping

File mới:

- `data_processing/paired_geometry.py`;
- `tests/test_paired_geometry.py`.

Quy trình TDD và kiểm tra:

1. Targeted test ban đầu thất bại đúng dự kiến với `ModuleNotFoundError` vì module chưa tồn tại.
2. Bản triển khai đầu tiên đạt 11 targeted tests và 58 full-suite tests.
3. Spec review: `APPROVED`.
4. Code-quality review phát hiện hai vấn đề: sai số gần biên có thể để tọa độ ngoài crop và polygon lõm có giao đa thành phần có thể sinh bridge giả.
5. Đã thêm regression tests và sửa: phân loại biên nghiêm ngặt, tọa độ intersection nằm đúng boundary, từ chối bounds không hữu hạn, và loại bảo thủ kết quả disconnected/self-touching thay vì ghi polygon YOLO sai.
6. Kết quả cuối: **17 targeted tests passed**; **64 full-suite tests passed**, 14 warnings cũ từ Matplotlib/PyParsing.
7. Code-quality re-review: `APPROVED`; không còn blocker của Task 1.

Không có dataset nào bị sửa hoặc ghi đè trong Task 1. Commit được hoãn tới bước đồng bộ repo Git sau khi hoàn thành nhóm task đã review.

#### Kết quả Task 2 — Reversible ViewTransform metadata

Đã bổ sung frozen dataclass `ViewTransform` và test riêng trong `tests/test_view_transform.py`.

Contract cuối cùng:

- lưu source size, crop box, nominal scale, kích thước raster thực sau resize, padding và canvas size;
- mapping dùng effective scale riêng cho trục x/y từ kích thước raster thực, khớp cách OpenCV làm tròn xuống;
- nominal scale phải sinh đúng `int(crop_dimension × scale)` và không được mâu thuẫn với resized dimensions;
- crop phải nằm trong source; resized raster cộng padding phải nằm trong canvas;
- serialization có `schema_version=1`, đọc được transform dict hoặc nested record và bỏ qua các record-level fields không liên quan;
- từ chối NaN/Inf, scale/canvas/crop không hợp lệ và schema version chưa hỗ trợ.

Quy trình review:

1. TDD red xác nhận `ViewTransform` chưa tồn tại.
2. Bản đầu đạt 15 targeted và 79 full-suite tests; spec review `APPROVED`.
3. Quality review phát hiện ba vấn đề về canvas containment, nominal/effective scale và schema cứng; đã sửa bằng actual resized dimensions và schema-aware parsing.
4. Re-review phát hiện nominal scale còn có thể mâu thuẫn raster; đã thêm floor-match invariant theo chính `letterbox_image` hiện tại.
5. Kết quả cuối: **24 targeted tests passed**, **88 full-suite tests passed** trong 58,39 giây; 14 dependency warnings cũ.
6. Code-quality re-review: `APPROVED`, không còn issue Task 2.

Không có dataset nào bị sửa trong Task 2. Bước tiếp theo là tích hợp clipping và metadata vào `01_preprocess.py` bằng test trước.

#### Kết quả Task 3 — Tích hợp preprocessing geometry/metadata

Các file thay đổi:

- `data_processing/01_preprocess.py`;
- `data_processing/paired_geometry.py`;
- `tests/test_data_pipeline.py`;
- `tests/test_paired_geometry.py`.

Chức năng đã triển khai:

- thay coordinatewise clamp bằng true polygon–rectangle clipping;
- label transformation dùng effective scale theo raster resize thực;
- sinh stable instance audit với source/intersection/canvas polygon, area, status và reason code;
- ghi `metadata/transforms.jsonl` deterministically bên trong transactional build;
- lưu exact artifact proxies: hair-mask coverage, vignette crop ratio, Gray-World gains và correction magnitude;
- thêm destination versioned an toàn `dataset_yolo_640x640_multiview_geom_v2`;
- thêm `--generate-v7-eval`, ghi val/test v7 vào các folder `_v7_eval` tách biệt và không đưa vào training YAML;
- giữ mặc định train v1+v7, val/test v1-only;
- backup transactional dùng tên UUID và chỉ xóa/restore backup do chính transaction hiện tại tạo.

Lịch sử TDD/review:

1. Bảy integration tests mới thất bại đúng dự kiến trước implementation.
2. Bản đầu đạt 98 full-suite tests, 1 skip; spec review `APPROVED`.
3. Quality review phát hiện backup cố định có thể xóa dữ liệu cũ và bbox inference gắn sai reason code; đã sửa bằng owned unique backup và `polygon_intersects_rect`.
4. Re-review phát hiện polygon nguồn suy biến trong crop bị gắn `OUTSIDE`; đã sửa: `OUTSIDE` chỉ áp dụng khi source area dương và thật sự không giao crop, còn polygon collapsed/duplicate là `DEGENERATE_AFTER_CLIP`.
5. Kết quả agent cuối: **110 passed, 1 skipped**, `py_compile` đạt. Test skip là kiểm tra tạo directory symlink do Windows không cấp quyền; static path-resolution logic vẫn được spec reviewer duyệt.
6. Spec review và code-quality re-review cuối đều `APPROVED`.

Không chạy preprocessing trên dataset thật trong Task 3 và không thay đổi hai processed dataset hiện có. Trước commit/push phải chạy verification độc lập trên bản sync Git.

Verification độc lập trên `D:\PAPER_SPKT\SPKT_PAPER_sync` trước checkpoint GitHub:

- `python -m py_compile` cho preprocessing, geometry và ba test module: exit code 0;
- `python -m pytest -q`: **110 passed, 1 skipped, 14 warnings** trong 9,72 giây;
- test skip duy nhất: directory-symlink capability trên Windows;
- `git diff --check`: exit code 0, không có whitespace error;
- trạng thái trước sync của repo Git sạch; chỉ bảy file source/test/docs thuộc Task 1–3 thay đổi sau sync.

Checkpoint GitHub Task 1–3:

- secret-pattern scan: không phát hiện token/API key/password rõ ràng;
- commit `9c9c582` — `feat: add auditable paired-view geometry preprocessing`;
- push thành công lên `origin/main` của `Hainguyen752004/SPKT_PAPER`;
- dataset ảnh, `runs`, cache và checkpoint huấn luyện không nằm trong commit.

### Bắt đầu Task 4 — Paired geometry publication gate

Mục tiêu: tạo auditor độc lập đọc dataset corrected và `metadata/transforms.jsonl`, sau đó quyết định pair nào đủ điều kiện cho AVC. Auditor không sửa ảnh/label và không tự động đưa pair lỗi vào training consistency.

Các invariant bắt buộc:

- mọi `(pair_id, view)` là duy nhất và đủ view cần thiết;
- label file khớp metadata instance/class/polygon;
- polygon hữu hạn, diện tích dương và nằm trong canvas;
- source→canvas→source round-trip error tối đa `1e-5` pixel;
- P2 mask được rasterize full canvas bằng OpenCV `fillPoly`, downsample stride 4 bằng max pooling và không rỗng;
- pair bị loại khỏi AVC có stable reason code, nhưng pair lỗi hợp lệ do crop không nhất thiết làm cả dataset gate thất bại;
- report JSON được ghi atomically và CLI exit khác 0 khi dataset-level gate thất bại.

Task 4 tiếp tục theo TDD và hai vòng review trước khi chạy trên dữ liệu thật.

#### Kết quả Task 4 — Auditor và publication gate

File mới:

- `data_processing/audit_paired_geometry.py`;
- `tests/test_paired_geometry_audit.py`.

Auditor hiện thực hiện:

- parse JSONL metadata và YOLO labels theo kiểu total/safe, không crash với JSON hợp lệ nhưng sai type;
- kiểm tra image/label/metadata existence, duplicate view/file, instance/class/source correspondence;
- tái tính source→crop clipping và crop→canvas transform;
- kiểm tra v1 phải dùng full-source crop;
- so khớp source path, source dimensions và preprocessing version giữa v1–v7;
- kiểm tra polygon label sau serialization có ít nhất ba điểm duy nhất và diện tích **lớn hơn 0**, không loại nhầm polygon rất nhỏ nhưng vẫn raster được;
- raster P2 đúng contract `cv2.fillPoly → adaptive_max_pool2d`, bảo toàn tiny positive polygon;
- giới hạn canvas trước allocation, decode ảnh thật và so kích thước ảnh với metadata;
- tách structural gate failures khỏi legitimate crop exclusions;
- đếm reason theo từng polygon, kể cả nhiều zero-area polygon trong cùng label file;
- ghi report JSON atomically, dọn temp và giữ report cũ nếu `os.replace` thất bại;
- CLI trả exit 0 khi gate pass, nonzero khi gate fail và 2 khi không ghi được output.

Lịch sử TDD/review chính:

1. Initial RED: module auditor chưa tồn tại; bản đầu đạt 12 focused tests.
2. Spec review phát hiện thiếu source→crop recomputation và malformed UTF-8 crash; bổ sung lên 22 focused tests, spec review `APPROVED`.
3. Quality review phát hiện unhashable JSON, zero-area label, v1 semantics, cross-split counts, canvas OOM risk, corrupt image và atomic cleanup; bổ sung adversarial tests và sửa.
4. Re-review tiếp tục phát hiện overflow từ số hữu hạn cực lớn, tiny-positive polygon bị loại nhầm và thiếu pair source/version checks; đã sửa.
5. Vòng cuối phát hiện nhiều zero-area polygon trong một file chỉ đếm một; parser hiện quét toàn bộ dòng và trả structured reason counter.
6. Kết quả agent cuối: **41 focused tests passed**, **151 full-suite tests passed, 1 skipped**, `py_compile` đạt.
7. Spec review và code-quality review cuối: `APPROVED`.

Không chạy auditor trên dataset thật trong Task 4. Bước kế tiếp là sync/push source đã review, sau đó mới sinh `geom_v2` và chạy gate trên toàn bộ 8.008 pair.

Verification độc lập Task 4 trên repo sync:

- `python -m py_compile data_processing/audit_paired_geometry.py tests/test_paired_geometry_audit.py`: exit code 0;
- `python -m pytest -q`: **151 passed, 1 skipped, 14 warnings** trong 7,56 giây;
- `git diff --check`: exit code 0;
- repo sạch trước sync; chỉ auditor, auditor tests, plan và `tailieu.md` thay đổi sau sync.

Checkpoint GitHub Task 4:

- secret-pattern scan: không phát hiện credential rõ ràng;
- commit `64b98b0` — `feat: add paired geometry publication gate`;
- push thành công lên `origin/main`;
- không có dataset, run hoặc checkpoint huấn luyện trong commit.

### Bắt đầu Task 5 — Sinh và audit dữ liệu geometry v2 thật

Trước khi chạy phải xác nhận destination versioned chưa tồn tại và dung lượng đĩa đủ. Lệnh dự kiến không dùng `--overwrite`:

```powershell
python data_processing/01_preprocess.py --destination data/dataset_yolo_640x640_multiview_geom_v2 --generate-v7-eval
```

Hai processed dataset cũ tiếp tục được giữ nguyên. Nếu destination v2 đã tồn tại, dừng để kiểm tra thay vì tự động xóa.

#### Lần chạy Task 5 số 1 — Dừng trước khi xử lý ảnh

Kiểm tra trước chạy:

- destination `geom_v2` chưa tồn tại;
- ổ D còn khoảng 117,1 GB trống.

Lệnh chạy dừng với exit code 1 tại dòng in console `Nguồn dữ liệu gốc`. Stack trace: `UnicodeEncodeError` từ `encodings/cp1252.py`; `sys.stdout.encoding` của môi trường là `cp1252`, không biểu diễn được chữ Việt. Kiểm tra source cho thấy đây là print Unicode duy nhất trong CLI; destination vẫn không tồn tại và không có build directory dở.

Root cause: nội dung log CLI giả định terminal UTF-8, trong khi Windows process hiện tại dùng CP1252. Đây là lỗi portability của console, không phải lỗi ảnh, label, OpenCV hoặc transactional build. Sẽ thêm regression test và thay log runtime bằng chuỗi ASCII/encoding-safe trước khi chạy lại.

Kết quả sửa console portability:

- TDD red: focused test thất bại đúng với `UnicodeEncodeError` ở nhãn tiếng Việt;
- thay startup label bằng ASCII `Source data:`; không reconfigure `sys.stdout` và không che lỗi encoding;
- focused green: 1 passed;
- full suite: **152 passed, 1 skipped**, 14 warnings;
- `py_compile` đạt;
- review độc lập: `APPROVED`, gồm kiểm tra subprocess dưới `PYTHONIOENCODING=cp1252`.

Destination v2 vẫn chưa tồn tại; đủ điều kiện chạy lại lần 2 mà không dùng `--overwrite`.

### Checkpoint sau cau hoi: lam gi tiep de ra ket qua nhanh

Nguoi dung hoi: "gio minh lam gi tiep theo de co the ra ket qua nhanh luon ta".

Lenh/kiem tra da thuc hien:

- Kiem tra thu muc goc `D:\PAPER_SPKT\Ham1000_p2_CBAM`: thay source/tests/docs/data/runs va cac file train/yaml hien co.
- Kiem tra process Python bang `Get-Process python` va `Get-CimInstance Win32_Process`: khong thay process `data_processing/01_preprocess.py` dang chay; cac process Python hien co chu yeu la `inference.web_app` va pgAdmin.
- Kiem tra `data`: co build tam `.dataset_yolo_640x640_multiview_geom_v2.build-54exvcl9`, chua co destination chinh thuc `dataset_yolo_640x640_multiview_geom_v2`.
- Dem build tam: 13.966 file. Ket luan: lan sinh `geom_v2` chua hoan tat thanh destination chinh thuc; khong duoc xem nhu dataset da san sang.
- Doc `SPKT_PAPER_sync/audit_reports/final_review_processed.json`: dataset hien co co source train/val/test = 8008/998/1007; `dataset_yolo_640x640_multiview` train/val/test = 16016/998/1007; `dataset_yolo_aug_p2_cbam` train/val/test = 31880/998/1007; skipped_files = 0; invalid_files = []; cross_split_overlaps = {}.
- Doc `dataset_p2_cbam.yaml`: hien dang tro vao `data/dataset_yolo_aug_p2_cbam`, dung de train nhanh.
- Doc `03_train_p2_cbam.py`: script train SkinSeg-YOLO26n P2-CBAM, mac dinh 300 epochs, batch Windows = 8, workers = 0, project `runs/segment`, name conference/test tuy so epoch.

Nhan dinh:

- Duong nhanh nhat de co bang ket qua hoi nghi la train truoc tren dataset da audit so luong `dataset_yolo_aug_p2_cbam`.
- `geom_v2` va auditor paired geometry van quan trong cho novelty/AVC, nhung vi output chinh thuc chua co nen khong nen cho no chan ket qua metric dau tien.
- De ra so nhanh: chay smoke 1-3 epoch de kiem tra moi truong/YOLO26, sau do chay training ngan 30-50 epoch cho bang hoi nghi tam thoi. Neu on thi tang epoch qua dem.

### Kiem tra moi truong train nhanh

Lenh/ket qua:

- `nvidia-smi`: co GPU NVIDIA GeForce RTX 4060 Laptop, VRAM 8.188 MiB; dang co process `vungcam_2026\python.exe` dung GPU.
- Kiem tra file weights/config: `models/yolo26n-seg.pt` va `models/yolo26n-seg-p2-cbam.yaml` deu ton tai.
- Lenh Python dau tien dung cu phap heredoc cua bash trong PowerShell nen fail parse truoc khi chay; day la loi lenh kiem tra, khong anh huong code/dataset.
- Chay lai bang here-string PowerShell voi `python` mac dinh thi import Ultralytics fail do base conda dang co NumPy 2.4.6 khong tuong thich binary OpenCV (`_ARRAY_API not found`, `numpy.core.multiarray failed to import`).

Ket luan:

- Khong dung base `python` de train.
- Can dung moi truong `vungcam_2026`: `conda run -n vungcam_2026 python ...`.

### Xac nhan moi truong `vungcam_2026`

Lenh:

```powershell
conda run -n vungcam_2026 python -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); print('ultralytics', ultralytics.__version__); from ultralytics.nn.modules import Segment26; print('Segment26', Segment26 is not None)"
```

Ket qua:

- torch `2.6.0+cu124`;
- CUDA available: `True`;
- GPU count: `1`;
- Ultralytics `8.4.13`;
- `Segment26` available: `True`.

Ket luan: co the train bang `conda run -n vungcam_2026 python 03_train_p2_cbam.py ...`.

### Smoke train 1 epoch P2-CBAM

Lenh:

```powershell
conda run -n vungcam_2026 python 03_train_p2_cbam.py --epochs 1
```

Trang thai sau 20 phut:

- Shell timeout 20 phut voi exit code 124, nhung process con `python 03_train_p2_cbam.py --epochs 1` van dang chay nen chua ket luan fail.
- Run dir da tao: `runs/segment/SkinSeg_YOLO26_P2_CBAM_Test`.
- File da co: `args.yaml`, `labels.jpg`, `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`, folder `weights`.
- `args.yaml`: epochs 1, batch 8, imgsz 640, device 0, workers 0, data runtime temp YAML tro vao `dataset_yolo_aug_p2_cbam`.
- `nvidia-smi`: process train PID 22440 dang dung GPU; VRAM tong khoang 5.374 MiB, GPU util co dao dong.

Nhan dinh:

- Model/dataset da qua duoc buoc khoi tao va ve batch, dang vao train thuc.
- Do Windows workers = 0 va dataset 31.880 anh segmentation, 1 epoch co the cham; can theo doi den khi co `results.csv` va weights.

### Theo doi smoke train

Sau khi shell timeout:

- Process `03_train_p2_cbam.py --epochs 1` van ton tai: wrapper PID 3904, child PID 22440.
- CPU time child PID 22440 khoang 2090 giay va van tang, working set khoang 306 MB; ket luan process dang lam viec, khong phai crash im lang.
- Thu muc `weights` chua co file; `results.csv` chua co. Voi Ultralytics, cac file nay thuong xuat hien sau khi epoch/validation ket thuc.

Tam thoi khong khoi dong job train thu hai de tranh tranh GPU/VRAM.

### Dung smoke train de chuyen sang duong nhanh hon

Ly do:

- Sau hon 20 phut, job 1 epoch full dataset van chua tao `results.csv`/weights.
- Process van dang chay, nen khong phai crash, nhung toc do nay khong phu hop voi muc tieu "ra ket qua nhanh luon".

Lenh dung job:

```powershell
$targets = @(3904,22440); foreach ($targetPid in $targets) { $procInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $targetPid" -ErrorAction SilentlyContinue; if ($procInfo -and $procInfo.CommandLine -like '*03_train_p2_cbam.py --epochs 1*') { Stop-Process -Id $targetPid -Force; "STOPPED $targetPid" } else { "SKIPPED $targetPid" } }
```

Ket qua:

- PID 3904 stopped.
- PID 22440 stopped.

Ghi chu loi lenh:

- Lenh dung dau tien dung bien `$pid`, bi PowerShell tu choi vi `$PID` la bien he thong read-only. Da doi thanh `$targetPid` va dung thanh cong.

De xuat tiep theo:

- Them option train nhanh vao `03_train_p2_cbam.py`: `--fraction`, `--batch`, `--workers`, `--name`.
- Chay sanity subset truoc, vi full 31.880 anh voi workers 0 tren Windows qua cham cho muc tieu lay ket qua ngay.

### Sua script train de chay nhanh

Muc tieu:

- Cho phep chay subset/fraction nho de lay ket qua pipeline nhanh.
- Giu default cu khi khong truyen option: epochs 300, batch Windows 8, workers Windows 0, name conference/test theo so epoch, fraction 1.0.

TDD:

1. Them test `test_train_cli_overrides_fast_run_controls` trong `tests/test_model_architecture.py`.
2. RED:

```powershell
conda run -n vungcam_2026 python -m pytest tests/test_model_architecture.py -k fast_run_controls -q
```

Ket qua RED: 1 failed, ly do dung mong doi: `argparse` bao `unrecognized arguments: --fraction 0.05 --batch 4 --workers 0 --name quick_smoke`.

Thay doi code:

- `03_train_p2_cbam.py` them option CLI:
  - `--fraction` trong khoang `(0, 1]`;
  - `--batch` so nguyen duong;
  - `--workers` so nguyen khong am;
  - `--name` de dat run name.
- Truyen cac option nay vao `model.train(...)`.

GREEN:

```powershell
conda run -n vungcam_2026 python -m pytest tests/test_model_architecture.py -k fast_run_controls -q
python -m py_compile 03_train_p2_cbam.py
```

Ket qua:

- focused test: 1 passed, 12 deselected;
- `py_compile`: exit code 0.

Ghi chu: sau focused pytest, conda/Windows co in `PermissionError` o `pytest-current` trong atexit, nhung command exit code 0 va test da pass.

### Thu chay quick smoke qua `conda run`

Lenh:

```powershell
conda run -n vungcam_2026 python 03_train_p2_cbam.py --epochs 1 --fraction 0.01 --batch 4 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1
```

Ket qua:

- Command tra exit code 0 nhung `conda run` in error report `UnicodeEncodeError` khi wrapper conda co gang print `response.stdout` ra console CP1252.
- Run folder `runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1` duoc tao.
- Co `args.yaml`, `labels.jpg`, `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`.
- Chua co `results.csv` va chua co weights.
- Khong con process `03_train_p2_cbam.py` sau lenh.

Ket luan:

- `conda run` khong phu hop de xem log train vi wrapper base conda gap loi encoding khi in stdout.
- Se chuyen sang goi truc tiep interpreter cua env:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 1 --fraction 0.01 --batch 4 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1_Direct
```

### Loi validation P2 mask va sua `mask_ratio`

Lenh direct-env:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 1 --fraction 0.01 --batch 4 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1_Direct
```

Ket qua:

- Train subset vao that: 319 train images, 998 val images, 80 train batches, GPU mem khoang 2.19G.
- Sau train epoch, validation crash:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x28224 and 112896x300)
```

Root cause:

- P2 segmentation head sinh mask/proto phan giai cao hon default validation mask ratio.
- Ultralytics default `mask_ratio=4` lam GT mask flatten 28.224 pixel, trong khi predicted mask flatten 112.896 pixel. Ti le dung bang 4 lan dien tich, tuong ung can `mask_ratio=2`.

TDD sua loi:

1. Them assertion vao `test_train_cli_overrides_fast_run_controls`: `train_calls[0]["mask_ratio"] == 2`.
2. RED:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -k fast_run_controls -q
```

Ket qua RED: 1 failed do `KeyError: 'mask_ratio'`.

3. Sua `03_train_p2_cbam.py`: truyen `mask_ratio=2` vao `model.train(...)`.
4. GREEN:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -k fast_run_controls -q
python -m py_compile 03_train_p2_cbam.py
```

Ket qua:

- focused test: 1 passed, 12 deselected;
- `py_compile`: exit code 0.

### Sua tuong thich Segment26 P2 voi validator

Quan sat sau khi dat `mask_ratio=2`:

- `args.yaml` da ghi `mask_ratio: 2`, nhung validation van crash cung shape mismatch.
- Doc source Ultralytics:
  - `SegmentationValidator._prepare_batch` resize GT mask ve `prepared_batch["imgsz"] // 4` khi dung `ops.process_mask`;
  - `SegmentationValidator.postprocess` tao predicted mask theo kich thuoc proto;
  - `Proto26` fuse feature dau vao va sau do `Proto` upsample, nen neu dua P2 vao proto thi proto ra stride 2.

Root cause dung:

- P2 prediction path hop le cho box/mask coefficient, nhung prototype branch cua stock validator can proto stride 4.
- Model P2-CBAM cu dua ca P2 vao `Proto26`, lam prediction masks co dien tich gap 4 lan GT masks trong validation.

TDD:

1. Them assertion vao `test_model_build_and_eval_forward_has_four_finite_scales`: voi input 256, `prediction_prototypes.shape[-2:] == (64, 64)`.
2. RED: focused `model_build` fail vi proto dang la `128x128`.
3. Sua `cbam.py`:
   - them `P2CompatibleSegment26`, subclass cua Ultralytics `Segment26`;
   - neu head co 4 scale thi detection/coefficients van dung P2-P5, nhung proto branch chi nhan P3-P5 de giu stride 4;
   - stock 3-scale Segment26 van giu nguyen behavior;
   - `register_cbam()` dang ky subclass nay duoi ten `Segment26` trong `ultralytics.nn.modules` va `ultralytics.nn.tasks`.
4. Loi trung gian: proto module da tao theo P3-P5 nhung forward van dua P2-P5, gay mismatch channel. Da override `forward()` de `Detect.forward(self, x)` dung du 4 scale, con `self.proto(...)` nhan `x[1:]` khi co 4 scale.
5. GREEN:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -k "model_build or fast_run_controls" -q
python -m py_compile cbam.py 03_train_p2_cbam.py
```

Ket qua:

- 2 passed, 11 deselected;
- `py_compile`: exit code 0.

### Quick smoke thanh cong sau P2 proto fix

Lenh:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 1 --fraction 0.01 --batch 4 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1_P2ProtoFix
```

Ket qua:

- Exit code 0.
- Train subset: 319 images, 80 batches.
- Val: 998 images, 998 instances, 125 batches.
- Model summary sau fix: `cbam.P2CompatibleSegment26`, 404 layers, 3.108.803 parameters, 14.0 GFLOPs.
- GPU mem train khoang 1.7G voi batch 4.
- Output: `results.csv`, `results.png`, confusion matrices, PR/F1/P/R curves, `weights/best.pt`, `weights/last.pt`.

Run folder:

```text
runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickSmoke_F001_E1_P2ProtoFix
```

Dong `results.csv` epoch 1:

```csv
epoch,time,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,train/sem_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M),val/box_loss,val/seg_loss,val/cls_loss,val/dfl_loss,val/sem_loss,lr/pg0,lr/pg1,lr/pg2
1,87.5608,3.89274,5.1963,40.3872,0.10638,6.2107,4e-05,0.01812,2e-05,1e-05,7e-05,0.0307,5e-05,1e-05,4.7059,5.02998,3.18339,0.1327,0,0.00071811,0.00071811,0.00071811
```

Nhan dinh:

- Day la smoke/sanity result, chua phai ket qua de bao cao trong paper vi chi 1 epoch va 1% train.
- Du de xac nhan pipeline YOLO26n P2-CBAM + dataset + validation segmentation da chay duoc.
- Buoc tiep theo de co so nhanh hon: chay 5 epoch voi `fraction=0.10`, batch 8 neu VRAM on.

### Quick result 5 epoch tren 10% train

Lenh:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 5 --fraction 0.10 --batch 8 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5
```

Ket qua:

- Exit code 0.
- Train subset: 3.188 images.
- Val: 998 images, 998 instances.
- Batch: 8.
- Model: `cbam.P2CompatibleSegment26`, 404 layers, 3.108.803 parameters, 14.0 GFLOPs.
- Hoan thanh 5 epochs trong 0.442 gio.
- Output:
  - `runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5/results.csv`;
  - `runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5/weights/best.pt`;
  - `runs/segment/SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5/weights/last.pt`;
  - plots/confusion/PR/F1/P/R curves va val prediction images.

Bang metric theo epoch tu `results.csv`:

| epoch | time_s | box_mAP50 | box_mAP50_95 | mask_mAP50 | mask_mAP50_95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 338.727 | 0.05535 | 0.02419 | 0.05691 | 0.03538 |
| 2 | 657.899 | 0.11509 | 0.05668 | 0.11258 | 0.07699 |
| 3 | 978.010 | 0.13321 | 0.06770 | 0.13300 | 0.09409 |
| 4 | 1313.530 | 0.15873 | 0.10521 | 0.15756 | 0.11571 |
| 5 | 1589.990 | 0.16915 | 0.10657 | 0.16338 | 0.11811 |

Final validation per class tu console:

| class | images | box_mAP50 | box_mAP50_95 | mask_mAP50 | mask_mAP50_95 |
|---|---:|---:|---:|---:|---:|
| all | 998 | 0.169 | 0.107 | 0.163 | 0.118 |
| akiec | 32 | 0.034 | 0.0171 | 0.0402 | 0.0196 |
| bcc | 51 | 0.101 | 0.0347 | 0.0649 | 0.0273 |
| bkl | 109 | 0.149 | 0.0884 | 0.149 | 0.0981 |
| df | 11 | 0.0433 | 0.0210 | 0.0310 | 0.0204 |
| mel | 111 | 0.116 | 0.0776 | 0.117 | 0.0881 |
| nv | 670 | 0.728 | 0.501 | 0.730 | 0.567 |
| vasc | 14 | 0.0116 | 0.00639 | 0.0116 | 0.00605 |

Nhan dinh:

- Day la ket qua nhap nhanh, chua phai final paper vi chi dung 10% train va 5 epoch.
- Pipeline da train/validate duoc, metric tang deu tu epoch 1 den 5.
- Neu can co bang hoi nghi trong hom nay: tiep theo nen chay full train 30-50 epoch qua dem hoac it nhat `fraction=1.0`, 10-20 epoch de lay baseline nghiem tuc.

### Sua Python 3.9 compatibility cho geometry tests

Khi chay full suite bang env train `vungcam_2026`:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest -q
```

Ket qua ban dau:

- collection error o `data_processing/paired_geometry.py`;
- Python 3.9 khong ho tro runtime annotation `float | None` neu thieu `from __future__ import annotations`;
- 3 collection errors: `test_paired_geometry.py`, `test_paired_geometry_audit.py`, `test_view_transform.py`.

Fix:

- Them `from __future__ import annotations` vao `data_processing/paired_geometry.py`.

Verification focused:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_paired_geometry.py tests/test_view_transform.py tests/test_paired_geometry_audit.py -q
python -m py_compile data_processing\paired_geometry.py
```

Ket qua:

- geometry/audit/view suite: 88 passed;
- `py_compile`: exit code 0.

### Sua Python 3.9 compatibility cho audit_dataset

Full suite trong env `vungcam_2026` sau fix `paired_geometry.py` con 1 failure:

```text
TypeError: zip() takes no keyword arguments
```

Vi tri:

- `data_processing/audit_dataset.py`, dong xu ly `dict(zip(SPLITS, values, strict=True))`.

Root cause:

- `zip(strict=True)` chi co tu Python 3.10, trong khi env train `vungcam_2026` dung Python 3.9.23.

Fix:

- Thay `zip(strict=True)` bang check do dai thu cong:
  - neu so gia tri `--expected-counts` khac so split thi raise `ValueError`;
  - neu dung thi `dict(zip(SPLITS, values))`.

Verification focused:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_data_pipeline.py -k report_path_safety -q
python -m py_compile data_processing\audit_dataset.py
```

Ket qua:

- focused test: 1 passed, 52 deselected;
- `py_compile`: exit code 0.

### Full verification sau cac fix train nhanh

Lenh:

```powershell
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest -q
```

Ket qua:

- exit code 0;
- 153 passed, 1 skipped trong 94.18 giay.

Ghi chu:

- Sau pytest van co `PermissionError` o `pytest-current` trong atexit cua Windows/conda, nhung command exit code 0 va test summary da pass.
- Khong dua thu muc `runs/` hoac weights vao GitHub; chi sync source/test/docs/log.

### GitHub sync cho fast training checkpoint

Repo sync:

```text
D:\PAPER_SPKT\SPKT_PAPER_sync
```

Verification truoc commit tren repo sync:

- `python -m py_compile 03_train_p2_cbam.py cbam.py data_processing\paired_geometry.py data_processing\audit_dataset.py tests\test_model_architecture.py`: exit code 0.
- `git diff --check`: exit code 0.
- `$env:PYTHONIOENCODING='utf-8'; & C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest -q`: 152 passed, 1 skipped trong 13.22 giay.
- Secret-pattern scan chi match cac dong log co chu `token/API key/password`, khong thay credential that.
- Staged scope: `03_train_p2_cbam.py`, `cbam.py`, `data_processing/audit_dataset.py`, `data_processing/paired_geometry.py`, `tests/test_model_architecture.py`, `README.md`, `tailieu.md`; khong stage `runs/`, weights moi hoac dataset.

Commit/push:

- commit `4a957ef` - `fix: enable fast P2-CBAM training validation`;
- push thanh cong len `origin/main`: `64b98b0..4a957ef`.

## 2026-08-11 - Requirements va quiet train log truoc khi chay full train

Yeu cau moi cua anh:

- Tao/kiem tra file `requirements.txt` de chuan bi train.
- Khi train khong in progress/log theo tung 1% hoac tung batch qua nhieu; chi can log gon theo epoch/summary de de doc.

Kiem tra ban dau:

- Thu muc `D:\PAPER_SPKT\Ham1000_p2_CBAM` chua co `requirements.txt`.
- Repo goc khong co `.codegraph`, nen khong dung CodeGraph.
- Env train that:
  - Python 3.9.23;
  - `torch==2.6.0+cu124`;
  - `torchvision==0.21.0+cu124`;
  - `ultralytics==8.4.13`;
  - CUDA runtime torch: 12.4;
  - `torch.cuda.is_available()` tra ve `True`.

Thay doi code train:

- Trong `03_train_p2_cbam.py`, them:

```python
os.environ.setdefault("YOLO_VERBOSE", "false")
```

- Dong nay duoc dat truoc `from ultralytics import YOLO` vi Ultralytics doc bien moi truong `YOLO_VERBOSE` ngay luc import.
- Trong `model.train(...)`, them:

```python
verbose=False
```

Tac dung:

- Tat TQDM/progress bar spam cua Ultralytics theo tung batch/phan tram.
- Console van co the in cac thong tin quan trong nhu model summary, epoch summary, validation/result summary.
- Muc tieu la train de doc hon, khong bi roi boi dong log 1%, 2%, 3%...

TDD cho quiet log:

RED:

```powershell
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -k "progress or fast_run_controls" -q
```

Ket qua RED:

- 2 failed:
  - chua co `os.environ.setdefault("YOLO_VERBOSE", "false")` truoc import Ultralytics;
  - `model.train(...)` chua truyen `verbose=False`.

GREEN sau khi sua:

```powershell
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -k "progress or fast_run_controls" -q
```

Ket qua:

- 2 passed, 12 deselected.

Verification requirements:

```powershell
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pip install --dry-run --no-deps -r requirements.txt
```

Ket qua:

- Exit code 0.
- Tat ca package trong `requirements.txt` deu `Requirement already satisfied` trong env `vungcam_2026`.

Import check:

```powershell
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -c "import torch, torchvision, ultralytics, cv2, numpy, albumentations, yaml, tqdm, matplotlib, pandas, PIL, scipy, pytest; print('requirements imports ok'); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
```

Ket qua:

- `requirements imports ok`;
- `torch 2.6.0+cu124 cuda 12.4 cuda_available True`.

Ghi chu:

- Khi import `albumentations`, thu vien nay tu check version online va co the in `ERROR:albumentations.check_version:Error fetching version info` do SSL cua Windows store. Lenh van exit code 0, khong phai loi train.
- Khi chay pytest tren Windows/conda, doi khi xuat hien `PermissionError` voi `pytest-current` trong atexit sau khi summary da pass. Neu exit code 0 va summary pass thi khong phai loi source.

Focused verification:

```powershell
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pytest tests/test_model_architecture.py -q
```

Ket qua:

- 14 passed.

Lenh train de anh chay tiep:

```powershell
cd D:\PAPER_SPKT\Ham1000_p2_CBAM
$env:PYTHONIOENCODING='utf-8'
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe 03_train_p2_cbam.py --epochs 30 --fraction 1.0 --batch 8 --workers 0 --name SkinSeg_YOLO26_P2_CBAM_Conference_E30
```

Neu anh muon cai moi package tu file moi:

```powershell
cd D:\PAPER_SPKT\Ham1000_p2_CBAM
& C:\Users\zinnn\miniconda3\envs\vungcam_2026\python.exe -m pip install -r requirements.txt
```

GitHub sync verification truoc commit:

- Sync sang repo `D:\PAPER_SPKT\SPKT_PAPER_sync` cac file:
  - `03_train_p2_cbam.py`;
  - `tests/test_model_architecture.py`;
  - `requirements.txt`;
  - `tailieu.md`.
- `python -m py_compile 03_train_p2_cbam.py tests\test_model_architecture.py`: exit code 0.
- `pytest tests/test_model_architecture.py -q`: 14 passed.
- `pip install --dry-run --no-deps -r requirements.txt`: exit code 0, tat ca package da satisfied.
- `git diff --check`: exit code 0.
- Commit/push len GitHub:
  - commit `b82365f` - `chore: add training requirements and quiet progress`;
  - push thanh cong len `origin/main`: `37f780e..b82365f`.

## 2026-08-11 - Ghi nhận hướng LiDAR Q1 và rà abstract hội nghị

### Tách riêng ý tưởng LiDAR

Đã tạo roadmap tương lai tại:

```text
D:\PAPER_SPKT\LiDAR_Q1_Roadmap.md
```

Hướng được ghi nhận là **uncertainty-calibrated domain generalization for LiDAR semantic segmentation under adverse weather and sensor degradation**. Roadmap này tách biệt với dự án HAM10000; không đổi hướng hoặc làm chậm bài hội nghị da liễu hiện tại.

### Trạng thái kết quả trước khi viết abstract

- Run `SkinSeg_YOLO26_P2_CBAM_QuickResult_F010_E5` chỉ dùng fraction 0,10 và 5 epoch; đây là quick/smoke evidence, không phải kết quả publication cuối.
- Run `SkinSeg_YOLO26_P2_CBAM_Conference_E30` hiện mới thấy `args.yaml`, chưa có `results.csv` trong lần rà này.
- Vì vậy chưa chèn mAP/Dice/FPS vào abstract. Nếu form tối nay là proposal abstract, dùng future tense `will be evaluated` là trung thực. Nếu là abstract của full paper, phải thay đoạn cuối bằng số liệu thực sau khi full train và test hoàn tất.

### Lỗi và điểm cần sửa trong bản abstract được gửi

1. `SkinSeg-YOLO26-P2Attn` không nói rõ attention là CBAM và không thống nhất với tên code P2-CBAM; dùng `SkinSeg-YOLO26-P2CBAM` hoặc bỏ tên riêng.
2. `vignette oriented cropping` nên đổi thành `vignetting-aware ROI cropping`.
3. Viết `images not belonging to the majority nevus (NV) class` để mô tả đúng NV-excluding image-level augmentation.
4. Không giữ các dấu nối do xuống dòng PDF như `preprocess-ing`, `Poly-gon`, `in-stance`.
5. `Index Terms—ttention mechanism` thiếu chữ A và `instance segment` chưa hoàn chỉnh.
6. Cần công khai nguồn/provenance của polygon trong Method; HAM10000 gốc là dataset classification nên không được để người đọc hiểu nhầm polygon là annotation gốc của HAM10000.
7. Với bản full paper, câu `will be evaluated` bắt buộc được thay bằng kết quả định lượng và kết luận chính.

### Abstract proposal-safe đã hiệu đính

> Skin lesion instance segmentation remains challenging because dermoscopic images frequently contain hair, dark peripheral borders, illumination variations, and fine lesion boundaries that may be degraded by feature downsampling. We present SkinSeg-YOLO26-P2CBAM, an Ultralytics YOLO26n-based framework that combines high-resolution feature prediction with artifact-oriented multi-view dermoscopic preprocessing. The standard P3–P5 segmentation pyramid is extended with a stride-4 P2 prediction branch, producing four-scale P2–P5 outputs. CBAM is inserted after the P2 and P3 backbone stages to recalibrate early channel and spatial features. During training, each source image is represented by a default letterboxed view and a deterministically processed view generated using vignetting-aware ROI cropping, DullRazor-inspired hair removal, Gray-World color constancy, and letterboxing. Polygon annotations are transformed consistently through the crop, resize, and padding operations. Additional geometric and photometric augmentation is applied only to images not belonging to the majority nevus (NV) class. The framework will be evaluated on a seven-class HAM10000-derived polygon dataset using controlled ablations of the P2 branch, CBAM, multi-view preprocessing, and NV-excluding augmentation. Evaluation will assess mask accuracy, boundary quality, class-wise performance, and computational cost.

Index terms đề xuất:

> **Index Terms—Attention mechanisms, dermoscopic imaging, instance segmentation, multi-view learning, skin lesion segmentation, YOLO26.**

Tên paper đang được ưu tiên:

> **Fine-Scale Skin Lesion Instance Segmentation via Multi-View Learning and a CBAM-Refined P2 Head in YOLO26**

## 2026-08-11 - Xác minh P2 head có tương thích với YOLO26 hay không

> [!IMPORTANT]
> **KẾT LUẬN KIẾN TRÚC:** P2 head hiện tại tương thích với YOLO26 đã pin. Model giữ `end2end=True`, `reg_max=1`, `C3k2`, `C2PSA` và `Segment26`; P2 không đưa DFL cũ trở lại và không biến model thành YOLO12.

> [!WARNING]
> **CÁCH CLAIM TRONG PAPER:** đây là custom YOLO26-based P2–P5 instance-segmentation adaptation. Không gọi nó là official YOLO26-P2 segmentation và không nói P2 trực tiếp tạo mask prototype.

### Kết luận

P2 head trong `models/yolo26n-seg-p2-cbam.yaml` **tương thích về kiến trúc và runtime với YOLO26 đang dùng**. Đây không phải việc mang nguyên head YOLOv8/YOLO12 sang YOLO26. Neck P2–P5 bám theo topology `yolo26-p2.yaml` chính thức của Ultralytics, sau đó đổi head cuối từ `Detect` sang biến thể `Segment26` hỗ trợ bốn scale.

### Đối chiếu với cấu hình YOLO26 cục bộ

Môi trường kiểm tra:

```text
ultralytics==8.4.13
```

Các file Ultralytics cục bộ xác nhận:

- `cfg/models/26/yolo26-p2.yaml` là cấu hình detection chính thức với output P2/P3/P4/P5;
- `cfg/models/26/yolo26-seg.yaml` là cấu hình segmentation chính thức với output P3/P4/P5;
- `Segment26` nhận số scale động từ `len(ch)`, nên detection/mask-coefficient branches có thể nhận bốn feature maps;
- `Proto26` hợp nhất nhiều scale nhưng mặc định lấy feature đầu vào đầu tiên làm mốc rồi upsample prototype.

Các đặc tính YOLO26 vẫn được giữ trong YAML tùy biến:

| Đặc tính YOLO26 | Trạng thái |
|---|---|
| `end2end: True` | Giữ nguyên; head có cả one-to-many và one-to-one branches |
| `reg_max: 1` | Giữ nguyên; `DFL` trở thành `Identity`, không đưa DFL cũ trở lại |
| `C3k2` backbone/neck | Giữ nguyên |
| `C2PSA` và `SPPF` | Giữ nguyên |
| `Segment26` | Giữ nguyên contract, mở rộng lên bốn prediction scales |
| P2–P5 strides | Đúng `[4, 8, 16, 32]` |

Việc thêm P2 chỉ thêm đường đặc trưng và prediction scale stride 4; nó không khôi phục các thành phần YOLO26 đã loại bỏ như distributional DFL khi `reg_max=1`, và không tắt end-to-end mode.

### Vì sao cần `P2CompatibleSegment26`

Nếu đưa thẳng bốn input P2–P5 vào `Proto26` mặc định, P2 sẽ trở thành feature mốc và lớp `Proto` tiếp tục upsample, tạo prototype ở stride 2. Điều này từng gây lệch kích thước giữa predicted masks và ground-truth masks.

`cbam.py` hiện xử lý có chủ đích:

```text
P2, P3, P4, P5 -> Detect/Segment26 prediction branches
P3, P4, P5     -> Proto26 -> prototype 160 x 160 khi input 640
```

Do đó:

- P2 vẫn tạo prediction locations và mask coefficients ở stride 4;
- prototype được tạo từ P3–P5 và upsample về stride 4;
- Với input 640, prototype native đã được kiểm tra là 160 × 160 (stride 4). `mask_ratio=2` hiện làm dataset chuẩn bị target mask ở độ phân giải cao hơn; loss Ultralytics 8.4.13 nội suy prototype khi target và prototype khác kích thước. Smoke train đã chạy thành công, nên đây không phải lỗi tương thích. Tuy nhiên `mask_ratio=2` là một hyperparameter huấn luyện riêng, không phải điều kiện để P2 hoạt động sau proto fix; phải giữ giống nhau giữa các ablation và ghi rõ trong Method. Có thể đánh giá lại `mask_ratio=4` trước thí nghiệm cuối nếu ưu tiên contract mặc định/tốc độ.

### Runtime evidence

Khởi tạo model và forward tensor `1 x 3 x 640 x 640` cho kết quả:

```text
head_class = P2CompatibleSegment26
nl = 4
stride = [4.0, 8.0, 16.0, 32.0]
reg_max = 1
end2end = True
dfl = Identity
head_from = [21, 24, 27, 30]
detections shape = (1, 300, 38)
prototype shape = (1, 32, 160, 160)
one2many feature scales = 4
one2one feature scales = 4
parameters = 3,108,803
```

### Cách viết chính xác trong paper

Được viết:

> The YOLO26 segmentation neck is extended with a stride-4 P2 prediction path, yielding four P2–P5 detection and mask-coefficient scales while retaining the end-to-end Segment26 formulation. Mask prototypes remain at stride 4 and are generated from fused P3–P5 features.

Không viết:

- “P2 restores DFL”;
- “P2 modifies the YOLO26 end-to-end mechanism”;
- “P2 directly generates the mask prototypes”;
- “P2 is an official YOLO26 segmentation model.”

Cách gọi đúng là **custom YOLO26-based P2–P5 instance-segmentation adaptation**. Ultralytics có official YOLO26-P2 detection topology, còn việc thích nghi nó sang `Segment26` bốn mức là phần triển khai của dự án.

### Ghi chú phiên bản

`requirements.txt` đang pin `ultralytics==8.4.13` và phiên bản này đã được xác minh có `Segment26`. Thông báo lỗi trong `03_train_p2_cbam.py` hiện gợi ý `8.4.60`; đây là một bất nhất câu thông báo cần sửa sau, nhưng không ảnh hưởng model đang chạy với dependency đã pin.

### Verification sau khi rà kiến trúc

Lần chạy pytest đầu tiên dùng temp mặc định của Windows:

```powershell
python -m pytest -p no:cacheprovider tests/test_model_architecture.py -q
```

Kết quả: 8 passed, 6 setup errors. Cả sáu error đều phát sinh trước test body tại `C:\Users\zinnn\AppData\Local\Temp\pytest-of-zinnn` với `PermissionError: [WinError 5]`; không có assertion kiến trúc nào fail.

Chạy lại với basetemp riêng nằm trong workspace:

```powershell
python -m pytest -p no:cacheprovider --basetemp D:\PAPER_SPKT\Ham1000_p2_CBAM\.pytest_p2_verify_20260811 tests/test_model_architecture.py -q
```

Kết quả xác nhận:

```text
14 passed in 6.96s
exit code 0
```

Kết hợp với forward 640 × 640 thành công và bốn stride đúng, không thấy compatibility blocker giữa P2 head tùy biến và YOLO26/Segment26 đã pin.

## 2026-08-11 - Phân tích Introduction của dự án LiDARGuard được cung cấp

File đã đọc đầy đủ:

```text
C:\Users\zinnn\.codex\attachments\6e114e03-4c69-4c52-b690-7d05333226f4\pasted-text.txt
```

### Dự án này thực chất làm gì?

> [!IMPORTANT]
> **LiDARGuard không phải một backbone LiDAR semantic segmentation mới.** Nó là một lớp giám sát độ tin cậy hậu xử lý gắn lên một RangeRet đã train và đóng băng. Mục tiêu là dự đoán khi nào kết quả segmentation của cả scan có khả năng sai, sau đó chỉ cấp thêm compute để sửa các scan rủi ro.

Luồng hệ thống:

```text
LiDAR scan
  -> frozen RangeRet
  -> pointwise semantic prediction
  -> global statistics + six-channel spatial risk map
  -> LiDARGuard quality/failure estimators
  -> scan-level mIoU estimate + classwise quality + risk ranking + failure probability
  -> fixed budget scheduler
  -> only low-confidence scans receive circular-shift TTA + range-space kNN refinement
```

Đầu vào Guard gồm hai nhóm:

1. **Global statistics:** maximum softmax probability, prediction margin, entropy, class composition, point density, range distribution, remission, occupancy và geometric–semantic boundary disagreement.
2. **Spatial representation:** risk map sáu kênh giữ vị trí uncertainty, range discontinuity, semantic boundary, boundary mismatch và valid observations.

Guard dùng statistics-anchored neural residual model: statistical anchors cho dự đoán tham chiếu dễ audit; neural residual bị giới hạn để cải thiện dự đoán nhưng không lệch vô hạn. Các model chuyên biệt dự đoán quality/ranking/failure, sau đó failure probability được calibration bằng temperature scaling và calibration-only rank-quantile transport. Ba Guard seed được ensemble nhưng dùng chung một segmentation backbone.

### Bài toán khoa học của LiDARGuard

Introduction lập luận rằng mIoU trên clean test set chưa đủ cho hệ thống an toàn. Khi gặp sparsity, missing returns, motion, weather, sensor artifact hoặc range-dependent sampling, model có thể dự đoán sai nhưng vẫn rất tự tin. LiDARGuard vì vậy giải quyết ba câu hỏi:

1. Chất lượng segmentation của toàn scan hiện tại ước lượng là bao nhiêu khi không có ground truth lúc deploy?
2. Scan này có nguy cơ failure hay không, và nên được xếp hạng rủi ro ở vị trí nào?
3. Với ngân sách compute cố định, scan nào đáng được chạy thêm TTA và kNN refinement?

Đây là hướng **post-hoc reliability monitoring + calibrated failure prediction + budgeted selective refinement**, khác với hướng `RiskLiDAR` đã ghi ở roadmap trước đó. `RiskLiDAR` tập trung huấn luyện chính segmentation backbone để robust/calibrated hơn; LiDARGuard giữ backbone frozen và xây guard/scheduler bên ngoài. Hai hướng có thể bổ sung nhau nhưng không được mô tả là cùng một contribution.

### Thiết kế thí nghiệm được nêu trong Introduction

- Dataset: SemanticPOSS.
- Split theo sequence cho train/calibration/test.
- Test: 500 base scans độc lập.
- Mỗi base scan có clean + 6 corruption families × 3 severity = 19 conditions; tổng 9.500 scan conditions.
- Hai corruption families được giữ hoàn toàn khỏi Guard training/calibration để đo unseen-corruption generalization.
- Ba seed, mười ablation, frozen statistical baselines.
- Clustered bootstrap theo base scan và Holm correction cho multiple endpoints.
- Đo latency, parameter count, throughput và peak memory.

Các kết quả được Introduction tuyên bố:

| Endpoint | LiDARGuard |
|---|---:|
| Scan-quality MAE | 0,0400 |
| R² | 0,8114 |
| Spearman correlation | 0,9003 |
| Failure AUROC | 0,9502 |
| Failure AUPR | 0,8225 |
| Brier score | 0,0631 |
| ECE | 0,0160 |

So với matched frozen statistical baselines, Introduction claim MAE giảm từ 0,0594 xuống 0,0400; failure AUPR tăng từ 0,7743 lên 0,8225; Brier giảm từ 0,0719 xuống 0,0631; bốn primary endpoints còn significant sau Holm correction với adjusted `p <= 0,0064`. Selective refinement được claim cải thiện mIoU có ý nghĩa thống kê ở budget 10–50%.

### Đánh giá ban đầu

Nếu code, split, raw predictions và bảng thống kê xác nhận đúng các con số trên, đây là một thiết kế paper mạnh hơn một bài chỉ thêm attention vào backbone. Điểm mạnh là bài toán vận hành rõ, backbone frozen, calibration/test tách biệt, compute budget rõ, nhiều corruption/severity, unseen families và thống kê theo independent base scan.

> [!WARNING]
> **INTRODUCTION KHÔNG ĐỦ ĐỂ XÁC NHẬN KẾT QUẢ HOẶC Q1.** Tất cả metric hiện chỉ là claim trong văn bản được cung cấp. Trước khi dùng hoặc đánh giá publication readiness phải kiểm tra code, manifest split, checkpoint, prediction files, corruption generator, seed outputs và script bootstrap/Holm.

### Các publication blocker/câu hỏi phản biện cần kiểm tra

1. **Target construction:** Guard train cần ground-truth scan mIoU/classwise IoU; phải chỉ rõ target được tạo ở train/calibration ra sao và không dùng test labels để fit.
2. **Leakage:** mọi corruption variant của cùng base scan phải ở cùng split; không được để clean scan ở train nhưng corruption của nó ở calibration/test.
3. **Sequence independence:** cần công bố sequence IDs cụ thể và chứng minh không overlap.
4. **Corruption protocol:** công thức, severity và random seed của cả sáu families phải tái lập; hai withheld families phải được chọn trước khi xem test result.
5. **Calibration isolation:** asymmetric projection, temperature và rank-quantile transport chỉ được fit/chọn trên calibration split rồi freeze.
6. **Failure definition:** threshold tạo nhãn failure phải được định nghĩa trước và sensitivity analysis cần cho thấy kết luận không phụ thuộc một threshold tiện lợi.
7. **Selective policy:** budget và low-confidence scheduler phải cố định từ calibration; TTA/kNN không được dùng test ground truth hoặc tune theo test.
8. **Baseline fairness:** statistical baselines và neural Guard phải dùng cùng input information, split, target và ensemble accounting.
9. **Compute accounting:** báo cả chi phí backbone, Guard, ensemble và phần refinement theo từng budget; không chỉ báo overhead Guard.
10. **Statistical unit:** bootstrap theo 500 base scans là hợp lý hơn bootstrap 9.500 conditions như thể độc lập; code phải đúng clustering này.
11. **Novelty search:** phải rà riêng scan-level quality prediction, LidarMetaSeg, selective inference, failure detection/calibration và corruption robustness trước khi claim mới.
12. **External validation:** chỉ SemanticPOSS có thể bị phản biện là hẹp; một cross-dataset hoặc backbone thứ hai sẽ làm hồ sơ Q1 mạnh hơn.

### Lỗi trình bày nhìn thấy ngay

- Dấu nối do xuống dòng PDF như `fun-damental`, `selec-tively`, `archi-tecture` phải được loại khi dùng source editable.
- Dòng `Corresponding author` đang chen giữa đoạn văn, cần đưa về footnote/template đúng vị trí.
- `Section ??` phải được thay bằng cross-reference hợp lệ trước khi compile bản cuối.
- Kiểm tra đủ bibliography [1]–[11], đúng tên RangeRet/Robo3D/LidarMetaSeg và đúng claim tương ứng.
- Các số liệu trong Introduction nên nhất quán tuyệt đối với Abstract, Results, tables và supplementary material.

### Kết luận định hướng

LiDARGuard là dự án **đánh giá và điều phối độ tin cậy ở runtime**, không phải dự án cải tiến độ chính xác backbone trực tiếp. Câu mô tả một dòng phù hợp:

> LiDARGuard predicts scan-level and classwise segmentation reliability from a frozen RangeRet model and selectively allocates test-time refinement to high-risk LiDAR scans under a fixed compute budget.

## 2026-08-11 - Tạo workspace riêng cho hướng IJCV LiDAR

Theo quyết định của tác giả, hướng LiDAR tương lai không làm theo LiDARGuard. Đã tạo workspace riêng:

```text
D:\PAPER_SPKT\LiDAR_OpenDomain_IJCV
```

Tài liệu trung tâm:

```text
D:\PAPER_SPKT\LiDAR_OpenDomain_IJCV\tailieu.md
```

Hướng làm việc được ghi nhận:

> **Sensor-aware cross-sensor, prompted open-vocabulary LiDAR-only 3D detection under compound domain and category shifts, using camera/VLM supervision only during training.**

Các quyết định chính:

- mục tiêu venue là IJCV, không phải ICCV;
- primary task là outdoor LiDAR 3D object detection;
- inference chỉ dùng LiDAR và text prompts;
- camera/VLM chỉ làm teacher trong training;
- kiến trúc thiên về one-stage range-view hoặc range/BEV hybrid để tận dụng thế mạnh YOLO/CV;
- gap cần kiểm chứng là source-only prompted seen/novel detection dưới compound cross-sensor/domain + category shift; không claim LiDAR-only VLM transfer tự nó là mới;
- chưa code cho đến khi systematic review, dataset audit và Gate 1 gap reproduction đạt.

Hai dự án được tách tuyệt đối. Việc tạo workspace Q1 không thay đổi code, dataset, train run hoặc experiment matrix của bài hội nghị HAM10000.

## 2026-08-11 - Hướng dẫn viết Introduction khi chưa có kết quả cuối

### Kết luận

Có thể và nên viết Introduction ngay khi chưa có kết quả full train. Introduction chủ yếu xác lập bối cảnh, vấn đề, khoảng trống, câu hỏi nghiên cứu, phương pháp và phạm vi đóng góp. Phần này không cần chờ bảng kết quả hoàn chỉnh. Tuy nhiên, mọi phát biểu về hiệu quả phải giữ ở mức giả thuyết, mục tiêu thiết kế hoặc kế hoạch đánh giá cho đến khi có baseline, ablation và kết quả test đủ tin cậy.

Dự án không hoàn toàn chưa có kết quả: smoke train 1 epoch trên 1% train và quick run 5 epoch trên 10% train đã xác nhận pipeline có thể train/validate. Quick run đạt box mAP50/mAP50-95 là 0,169/0,107 và mask mAP50/mAP50-95 là 0,163/0,118 trên validation. Các số này chỉ là sanity evidence, không phải kết quả publication vì dùng một phần dữ liệu, số epoch thấp, chưa có baseline tương ứng và chưa có ablation.

### Cấu trúc Introduction đề xuất

Introduction có thể triển khai thành năm đoạn:

1. **Bối cảnh và ý nghĩa:** trình bày vai trò của phân đoạn thực thể tổn thương da trên ảnh dermoscopic. Không gọi bài toán là phân đoạn tế bào ung thư; đối tượng là vùng tổn thương da và polygon mang một trong bảy nhãn chẩn đoán.
2. **Khó khăn thực tế:** nêu lông, vignetting/viền tối, biến thiên ánh sáng và màu; tổn thương nhỏ hoặc biên khó; mất chi tiết sau downsampling; và phân bố chẩn đoán mất cân bằng của HAM10000.
3. **Khoảng trống nghiên cứu:** các head P3-P5 có thể thiếu thông tin không gian stride-4; chỉ dùng ảnh mặc định hoặc chỉ dùng ảnh đã xử lý artifact có thể không khai thác hai biểu diễn bổ trợ; hiệu quả của CBAM trên đặc trưng P2/P3 trong cấu hình YOLO26 segmentation này cần được đánh giá có kiểm soát.
4. **Phương pháp và câu hỏi nghiên cứu:** giới thiệu custom YOLO26-based P2-P5 instance-segmentation adaptation, CBAM tại P2/P3, hai train view v1/v7 và NV-excluding image-level augmentation. Đặt câu hỏi liệu các thành phần này có cải thiện phân đoạn, đặc biệt ở tổn thương nhỏ hoặc biên khó, thay vì khẳng định trước rằng chúng có hiệu quả.
5. **Đóng góp và tổ chức bài:** nêu đóng góp ở mức thiết kế, tích hợp và protocol đánh giá. Chỉ thêm một câu tóm tắt kết quả định lượng sau khi full experiment hoàn tất.

### Ngôn ngữ an toàn trước khi có full experiment

Được dùng:

- `we propose`, `we present`, `we investigate`;
- `is designed to preserve high-resolution information`;
- `aims to improve` hoặc `is intended to reduce`;
- `we define a controlled evaluation protocol`;
- `will be evaluated` nếu văn bản là proposal abstract hoặc proposal paper;
- `we hypothesize that` khi nêu tác dụng dự kiến.

Không được dùng trước khi có bằng chứng:

- `the proposed method improves/outperforms`;
- `significantly improves small-lesion segmentation`;
- `is robust to artifacts`;
- `solves class imbalance` hoặc `guarantees class balance`;
- `state of the art`, `the first`, `real-time`;
- mAP, Dice, IoU, FPS hoặc tỷ lệ phần trăm từ quick run như kết quả chính thức.

Nếu là abstract proposal, có thể dùng future tense. Nếu là abstract của full paper, các câu `will be evaluated` phải được thay bằng metric và kết luận chính đã được xác minh.

### Contribution paragraph proposal-safe

> The main contributions of this study are summarized as follows. First, we adapt the YOLO26 segmentation architecture to incorporate a stride-4 P2 prediction path, resulting in four-scale P2-P5 lesion instance segmentation. Second, CBAM is placed at the P2 and P3 stages to recalibrate high-resolution channel and spatial features. Third, we develop an artifact-oriented fixed multi-view training pipeline that combines a default letterboxed view with a deterministically processed dermoscopic view while consistently transforming polygon annotations. Finally, we establish a controlled ablation protocol to assess the individual and combined effects of the P2 branch, CBAM, multi-view preprocessing, and NV-excluding augmentation.

Đoạn trên chỉ tuyên bố việc thiết kế, tích hợp và xác lập protocol; không tuyên bố từng module là phát minh mới hoặc hệ thống đã tốt hơn baseline.

### Ranh giới contribution hiện tại

- Được mô tả là đã triển khai: YOLO26n `C3k2`/`C2PSA`, prediction scales P2-P5, CBAM tại P2/P3, `P2CompatibleSegment26`, pipeline v1/v7, polygon transformation, NV-excluding augmentation và các công cụ audit hiện có.
- Chưa được mô tả là contribution hoàn tất: Paired Artifact-View Consistency loss và P2 Boundary Supervision. Đây là thiết kế mở rộng trong roadmap; chỉ đưa vào paper như phương pháp đã thực hiện sau khi code, test và train tương ứng tồn tại.
- Không claim mới ở cấp thành phần riêng lẻ: P2, CBAM, DullRazor-inspired hair removal, Gray-World, HAM10000 hoặc multiscale segmentation đều có tiền lệ. Khoảng đóng góp có thể bảo vệ nằm ở cách tích hợp và đánh giá có kiểm soát trong bài toán cụ thể.

### Các điểm phải xử lý trước bản paper cuối

1. Thống nhất phiên bản Ultralytics: môi trường và `requirements.txt` dùng `8.4.13`, trong khi một đoạn trong `README.md` còn ghi `8.4.60`.
2. Công khai provenance của polygon. HAM10000 gốc là dataset classification; không để người đọc hiểu polygon là annotation gốc của HAM10000.
3. Bổ sung metadata bệnh nhân/ca bệnh và patient-level grouped-split audit. Kiểm tra overlap theo normalized ISIC image ID hiện tại chưa chứng minh zero patient-level leakage.
4. Chạy official YOLO26n-seg baseline và các ablation P2, CBAM, multi-view, augmentation và full model dưới cùng split, seed, epoch budget và quy tắc chọn checkpoint.
5. Báo cáo kết quả theo lớp, kích thước tổn thương, nhóm artifact, nhiều seed và chi phí tính toán; không chỉ báo metric all-class của một run.
6. Giữ nội dung LiDAR ở workspace riêng. Các ghi chú LiDARGuard/IJCV hiện có trong file này là nhật ký lịch sử, không được trộn vào narrative hoặc contribution của paper HAM10000.

### Quy tắc cập nhật sau khi có kết quả

Sau khi full train và ablation hoàn thành, quay lại sửa Introduction theo ba bước:

1. Thay ngôn ngữ mục tiêu như `aims to` bằng kết luận chỉ ở những điểm được kết quả hỗ trợ.
2. Thêm một câu kết quả ngắn ở cuối Introduction, nhất quán tuyệt đối với Abstract, Results, bảng và supplementary material.
3. Nếu một thành phần không cải thiện hoặc chỉ cải thiện một subgroup, mô tả đúng hiệu ứng quan sát được; không giữ claim tổng quát ban đầu.

## Ghi chú kỹ thuật ngày 2026-08-17: khóa checkpoint và chính sách final test

CBAM trong cấu hình hiện tại gồm channel attention và spatial attention theo công thức chuẩn, chỉ nhân lại trọng số chú ý nên giữ nguyên shape tensor. Nhánh dự đoán P2-P5 tương ứng stride 4/8/16/32; phần prototype của `P2CompatibleSegment26` dùng đầu vào P3-P5 nhưng vẫn sinh mask prototype ở stride 4. Run 200 epoch đã chứng minh graph forward/loss hoạt động ổn định đủ để đánh giá checkpoint, không phải chỉ là cấu hình trên giấy.

YAML hiện đúng cho scale `n`, nhưng các lớp CBAM đang ghi cứng channel 64/128. Điều này không tự an toàn khi đổi sang scale `s/m/l/x`; nếu mở rộng scale phải rà lại channel tại vị trí đặt CBAM trước khi train.

Checkpoint nên dùng là `best.pt`, vì best validation xuất hiện khoảng epoch 183 với mask mAP50 0.8664, mask mAP50-95 0.6874 và box mAP50-95 0.7118. Epoch 200 thấp hơn khoảng 0.0017 ở mask mAP50-95, đồng thời validation segmentation loss tăng nhẹ cuối run, nên dùng checkpoint tốt nhất theo validation thay vì checkpoint cuối.

Confusion hiện cho thấy `nv/bcc/df/vasc` mạnh hơn, còn `mel/akiec` yếu hơn. Rủi ro chính cần báo cáo thẳng là `mel->nv` và `akiec->bkl/background`.

Số polygon theo lớp là: nv 24138, mel 9336, bkl 9220, bcc 4316, akiec 2742, vasc 1188, df 966. Chiến lược augmentation loại trừ NV giúp không phóng đại lớp đa số, nhưng chưa cân bằng các minority class với nhau; `vasc/df/akiec` vẫn ít hơn rõ rệt so với `mel/bkl`.

Các biến gây nhiễu cần ablate gồm `optimizer=auto`, rotation 180 degrees, augmentation online cộng offline, mixup và copy-paste. Đối chứng hợp lý là AdamW với lr 0.001 và một profile augmentation dermoscopy ít tạo ảnh da không thực hơn.

Kết quả baseline PDF test ba seed 0.5636±0.0234 mask mAP50-95 không được so trực tiếp với validation một seed 0.6874 trên 31,880 train images. Hai con số khác split, khác số seed và khác điều kiện báo cáo nên chỉ dùng để định hướng, không dùng làm kết luận cải thiện trực tiếp.

Ma trận ablation cần khóa trước khi mở final test: A=YOLO26n baseline không P2/CBAM; B=P2-only; C=CBAM-only; D=P2+CBAM. Tất cả phải dùng cùng split, data, seed, hyperparameter và quy tắc chọn checkpoint bằng validation.

Quy tắc cuối: model và hyperparameter chỉ được chọn bằng validation only; test is final reporting only và không được dùng để chỉnh model, chọn A-D, chọn vị trí attention, chọn augmentation hoặc chọn hyperparameter. Nếu sau khi xem test mà vẫn muốn tiếp tục chọn model, phải giữ một final test set khác chưa từng mở; nếu không có, mọi phát triển tiếp theo chỉ dùng validation và kết quả test này chỉ là báo cáo checkpoint hiện tại.

## Kết quả benchmark test ngày 2026-08-17: P2-CBAM best.pt

Đã chạy script `04_evaluate_best_test.py` trên checkpoint:

```text
D:\PAPER_SPKT\SkinSeg_YOLO26_P2_CBAM_640\weights\best.pt
```

Kết quả được lưu tại:

```text
D:\PAPER_SPKT\Ham1000_p2_CBAM\runs\segment\SkinSeg_YOLO26_P2_CBAM_Test_Final_20260817\test_metrics.json
```

Thiết lập đánh giá: split `test`, image size 640, batch 16, device GPU 0, seed 0, Ultralytics 8.4.13, PyTorch 2.6.0+cu124, CUDA 12.4. Checkpoint SHA-256 là `0dbbb473ff03c210421b5f54918aadfd38ccfaffd786b48ba4e2ce068e5629c2`. Không thiếu metric nào trong tám metric chính của Ultralytics.

Kết quả tổng thể:

| Nhóm | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Box | 0.8767 | 0.8202 | 0.9060 | 0.7247 |
| Mask | 0.8696 | 0.8134 | 0.9006 | 0.7154 |

Nhận xét nhanh:

1. Kết quả test cao hơn validation best trước đó. Validation best mask mAP50-95 khoảng 0.6874, còn test đạt 0.7154. Chênh lệch này là tín hiệu tốt, nhưng chưa nên diễn giải là mô hình tổng quát hóa chắc chắn tốt hơn mọi baseline vì hiện mới có một checkpoint, một seed và protocol dữ liệu đã khác baseline PDF.
2. Box và mask khá sát nhau. Box mAP50-95 là 0.7247, mask mAP50-95 là 0.7154, chỉ thấp hơn khoảng 0.0093. Điều này cho thấy head segmentation không bị tụt quá nhiều so với localization; biên mask đang theo được vùng tổn thương tương đối ổn.
3. Precision cao hơn recall ở cả box và mask. Với mask, precision 0.8696 và recall 0.8134. Mô hình có xu hướng dự đoán khá chắc, ít false positive hơn, nhưng vẫn còn bỏ sót một phần lesion hoặc một số ca khó. Khi viết paper nên nói mô hình đạt độ chính xác cao nhưng recall vẫn là điểm cần theo dõi, đặc biệt ở `mel/akiec`.
4. Mask mAP50 đạt 0.9006 là rất tốt cho ngưỡng IoU 0.5. mAP50-95 đạt 0.7154 cho thấy khi tăng yêu cầu khớp biên từ dễ đến khó, mô hình vẫn giữ chất lượng tốt, nhưng vẫn còn khoảng cách giữa nhận diện vùng tổn thương và khớp biên thật chính xác.
5. So với baseline PDF đã báo cáo mask mAP50-95 0.5636±0.0234 trên test ba seed, con số 0.7154 là rất khả quan. Tuy nhiên, so sánh này chỉ nên viết là "tham chiếu định hướng" hoặc "preliminary comparison", chưa phải kết luận vượt trội cuối cùng, vì P2-CBAM hiện dùng dữ liệu augmentation lớn hơn và chưa có cùng số seed/ablation khép kín.

Kết luận tạm thời: checkpoint P2-CBAM `best.pt` có kết quả test mạnh, đặc biệt ở mask mAP50-95 0.7154 và mask mAP50 0.9006. Đây là đủ tốt để đưa vào phần kết quả sơ bộ và làm căn cứ viết Results/Discussion, nhưng kết luận khoa học cuối cùng vẫn cần ma trận ablation A-D cùng protocol thống nhất.

## Ghi chú đọc 3 paper nền ngày 2026-08-17: hướng sửa block P2-CBAM-v2

Mục tiêu của ghi chú này là giữ lại ngữ cảnh trước khi sửa model. Hiện chưa thay đổi YAML hoặc code kiến trúc; đây chỉ là phân tích định hướng để tránh quên lý do thiết kế.

### 1. Bài baseline YOLO26 của nhóm

Bài baseline YOLO26 là mốc protocol quan trọng nhất. Kết luận chính của bài này là không nên đánh đổi lung tung bằng preprocessing hoặc augmentation quá mạnh. Khi architecture, resolution, training budget và protocol đánh giá được giữ cố định, raw/minimally processed input với online augmentation cho trade-off tốt hơn các pipeline deterministic preprocessing phức tạp.

Điểm yếu còn lại cần nhắm tới là `mel -> nv`, `akiec -> bkl/background`, và minority recall. Vì vậy nếu sửa model thì không nên chỉ tăng tham số hay tăng neuron chung chung. Thay đổi phải nhắm vào class khó, boundary detail và context sau khi feature được fuse, vì nhiều lỗi không phải do mask overlap quá thấp mà do phân biệt chẩn đoán giữa các class có hình thái gần nhau.

### 2. Bài Scientific Reports dual-branch

Đây là bài đáng học nhất về block/kiến trúc. Các thành phần họ dùng gồm:

- ASPP để lấy multi-scale context bằng dilated convolution.
- Transformer để thêm global context.
- Multi-scale fusion với kernel 3x3, 5x5, 7x7.
- Attention Gate để lọc skip/fusion feature theo vùng liên quan.
- SE block để recalibrate channel.
- Mask/morphology branch để khai thác thông tin hình dạng, biên, bất đối xứng và độ phức tạp của lesion.

Tuy nhiên, mô hình của họ rất nặng, khoảng 66.3M parameters. Không nên bê nguyên EfficientNet-B7, Transformer đầy đủ, hoặc dual-branch morphology classifier vào YOLO26n vì sẽ phá mục tiêu lightweight và làm khó ablation. Cái đáng học là ý tưởng nhẹ: ASPP-lite hoặc DilatedContext cho tầng sâu, gated/adaptive fusion ở head, và SE/CBAM đặt đúng vị trí fusion thay vì chỉ tăng backbone.

### 3. Bài YOLOv11

Bài YOLOv11 chủ yếu là YOLOv11s-seg kết hợp app deployment. Metric được báo cáo gồm detection mAP50-95 khoảng 0.735 và segmentation mAP50-95 khoảng 0.706. Model P2-CBAM hiện tại của mình đạt mask mAP50-95 0.7154 trên test benchmark đã khóa, tức nhỉnh hơn con số segmentation mAP50-95 của bài YOLOv11.

Bài YOLOv11 hữu ích để đối chiếu kết quả và narrative ứng dụng, nhưng không cung cấp block mới mạnh bằng bài Scientific Reports. Do đó không nên lấy nó làm nguồn chính cho sửa kiến trúc.

### Hướng đề xuất cho model mới

Tạo model mới tên tạm `P2-CBAM-v2`, không sửa đè model cũ. Model cũ đang là mốc đã benchmark và đã push GitHub; mọi thử nghiệm mới phải giữ nó nguyên vẹn để còn so sánh.

Các hướng block có cơ sở:

1. Thêm `ASPP-lite` hoặc `DilatedContext` ở tầng sâu P5/P4 trước khi fusion xuống P3/P2. Mục tiêu là tăng receptive field và global/local context cho các lesion có hình thái dễ nhầm, nhưng không làm nặng như Transformer đầy đủ.
2. Thêm `GatedFusion` sau concat ở P2/P3 để học tỷ lệ lấy thông tin từ skip feature và upsample feature. Mục tiêu là tránh fusion kiểu concat cứng, giúp head tự chọn feature hữu ích cho boundary và class khó.
3. Giữ CBAM nhưng cân nhắc chuyển hoặc nhân đôi CBAM ở head fusion thay vì chỉ đặt trong backbone. Lý do là lỗi hiện tại nằm nhiều ở nhận diện class và boundary sau khi fuse feature, không chỉ ở feature extraction đầu vào.
4. Chưa nên thêm Transformer đầy đủ. Nếu cần global context, chỉ thử block nhẹ ở P5, không đặt ở P2 vì P2 có spatial resolution cao và attention đầy đủ sẽ rất tốn tài nguyên.

Ma trận ưu tiên thử nghiệm:

| Phiên bản | Thay đổi chính | Mục tiêu | Rủi ro |
|---|---|---|---|
| v2A | P2-CBAM + ASPP-lite ở P5 | Tăng context sâu, cải thiện mask mAP50-95 | Có thể chưa tác động đủ tới P2 boundary |
| v2B | P2-CBAM + GatedFusion ở P2/P3 | Cải thiện fusion, boundary và minority recall | Cần code custom module ổn định |
| v2C | P2-CBAM + ASPP-lite + GatedFusion | Kết hợp context sâu và fusion thích nghi | Nặng hơn, cần ablation rõ để biết module nào có ích |

Khuyến nghị hiện tại: bắt đầu với `v2B` hoặc `v2C`. Nếu ưu tiên an toàn và ít thay đổi, chọn `v2B`. Nếu ưu tiên khả năng tăng mAP cao hơn và chấp nhận thêm rủi ro, chọn `v2C`. Dù chọn hướng nào, phải tạo YAML mới và module mới, không sửa đè `yolo26n-seg-p2-cbam.yaml`.
## Ghi chú triển khai P2-CBAM-v2B GatedFusion ngày 2026-08-17

Đã triển khai nhánh kiến trúc `v2B` theo hướng an toàn: giữ nguyên model baseline `models/yolo26n-seg-p2-cbam.yaml` và tạo model mới riêng tại:

```text
models/yolo26n-seg-p2-cbam-v2b-gatedfusion.yaml
```

Thay đổi chính của v2B là thêm module `GatedFusion` trong file riêng `cbam_v2b.py`, còn `cbam.py` giữ vai trò baseline P2-CBAM. Module này nhận tensor đã concat, học gate theo channel và spatial, sau đó nhân lại vào feature map nhưng không đổi shape. Vì vậy nó phù hợp để đặt sau các điểm fusion của neck mà không phá contract của YOLO26/Segment26.

Vị trí đặt gate:

1. Sau concat P3 `Concat([-1, 6])`, trước `C3k2` P3.
2. Sau concat P2 `Concat([-1, 3])`, trước `C3k2` P2.

Do thêm hai layer mới, chỉ số head của v2B thay đổi so với baseline:

```text
Baseline Segment26 inputs: [21, 24, 27, 30]
v2B Segment26 inputs:      [23, 26, 29, 32]
```

Các kiểm tra đã chạy:

```text
python -m pytest tests\test_v2b_gated_fusion.py -q
python -m pytest tests\test_model_architecture.py -q
python -m py_compile cbam.py cbam_v2b.py 03_train_p2_cbam.py tests\test_v2b_gated_fusion.py tests\test_model_architecture.py
```

Kết quả xác minh:

- `tests/test_v2b_gated_fusion.py`: 3 passed.
- `tests/test_model_architecture.py`: 15 passed.
- `py_compile`: passed.

Script train `03_train_p2_cbam.py` đã có option chọn kiến trúc và optimizer rõ ràng:

```text
python 03_train_p2_cbam.py --architecture v2b --optimizer AdamW --name SkinSeg_YOLO26_P2_CBAM_v2B_GatedFusion_AdamW
```

Sau khi chốt protocol v2B ngày 2026-08-17, cấu hình train chính được đặt thành **200 epochs + AdamW** qua entry-point riêng:

```text
python 03_train_p2_cbam_v2b.py
```

Lệnh trên tương đương:

```text
python 03_train_p2_cbam.py --architecture v2b --optimizer AdamW --epochs 200 --name SkinSeg_YOLO26_P2_CBAM_v2B_GatedFusion_AdamW_E200
```

Nếu không truyền `--optimizer`, Ultralytics dùng mặc định `optimizer=auto`. Với Ultralytics 8.4.13, `auto` không đồng nghĩa luôn luôn là AdamW: run ngắn có thể chọn AdamW, nhưng run dài trên 10000 iterations sẽ tự chọn MuSGD. Vì vậy nếu muốn ablation AdamW sạch thì phải truyền `--optimizer AdamW` tường minh.

Ý nghĩa research: v2B là thử nghiệm “fusion-aware attention”, nhắm trực tiếp vào chỗ skip feature P2/P3 và semantic upsample feature trộn với nhau. Hướng này phù hợp với phân tích trước đó rằng model hiện đã localize/mask khá tốt, nhưng vẫn cần cải thiện class khó, minority recall và boundary/context sau fusion. Chưa dùng kết quả test final để chọn hyperparameter; v2B phải được train/so sánh bằng validation protocol trước, rồi mới quyết định có đáng đưa vào ablation chính hay không.

## Ghi chú claim từ A đến Z cho bản hội nghị SkinSeg-YOLO26-P2Attn ngày 2026-08-17

Phạm vi section này là bản hội nghị hiện tại, không bao gồm v2A/v2B/v2C. Thư mục kết quả khóa là:

```text
D:\PAPER_SPKT\SkinSeg_YOLO26_P2_CBAM_640_final
```

Trong thư mục này đã có đầy đủ `args.yaml`, `results.csv`, `results.png`, PR/F1/P/R curves cho box và mask, confusion matrix, batch visualization, `weights/best.pt`, `weights/last.pt`, và test final tại:

```text
D:\PAPER_SPKT\SkinSeg_YOLO26_P2_CBAM_640_final\SkinSeg_YOLO26_P2_CBAM_Test_Final_20260817\test_metrics.json
```

### 1. Tên model và phạm vi claim

Tên phù hợp cho hội nghị: **SkinSeg-YOLO26-P2Attn**.

Claim nên dùng:

- Mô hình là một framework dựa trên Ultralytics YOLO26n cho skin lesion instance segmentation.
- Mô hình mở rộng prediction pyramid thành bốn mức P2/P3/P4/P5, trong đó P2 có stride 4 để hỗ trợ chi tiết biên và tổn thương nhỏ.
- CBAM được chèn tại các stage P2/P3 của backbone để refine feature sớm có độ phân giải cao.
- Pipeline train dùng hai view cho mỗi ảnh nguồn: letterboxed view và artifact-processed view.
- Artifact-processed view gồm vignette-oriented cropping, hair removal lấy cảm hứng từ DullRazor, Gray-World color constancy và letterboxing.
- Polygon annotation được biến đổi nhất quán theo crop/resize.
- Offline augmentation bổ sung chỉ áp dụng cho ảnh không chứa class đa số NV.

Không nên claim:

- Không nói đây là YOLO26 official P2 segmentation model. Cách đúng là custom YOLO26n-based P2-P5 segmentation adaptation.
- Không nói CBAM, P2 hay DullRazor-inspired preprocessing là phát minh mới.
- Không nói SOTA tuyệt đối.
- Không nói mô hình phát hiện “tế bào ung thư”; đây là lesion instance segmentation/class-aware lesion prediction trên ảnh dermoscopic.
- Không dùng kết quả test final để biện minh chọn thêm v2A/v2B/v2C cho hội nghị. Các biến thể đó để Q1/future work.

### 2. Kết quả train/validation nên báo cáo

Run final train 200 epoch. Checkpoint tốt nhất được chọn theo validation, không chọn theo test.

Best validation theo `metrics/mAP50-95(M)` xuất hiện ở epoch 183:

| Split | Epoch | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation best | 183 | 0.8458 | 0.7835 | 0.8658 | 0.7118 | 0.8474 | 0.7850 | 0.8664 | 0.6874 |
| Validation epoch 200 | 200 | 0.8624 | 0.7804 | 0.8662 | 0.7109 | 0.8584 | 0.7772 | 0.8595 | 0.6857 |

Cách diễn giải:

- Validation đã hội tụ ổn quanh cuối training; epoch 183 là best checkpoint theo mask mAP50-95.
- Epoch 200 không sụp, nhưng mask mAP50-95 thấp hơn best khoảng 0.0017, nên dùng `best.pt` là đúng protocol.
- Precision cao hơn recall, tức model dự đoán tương đối chắc nhưng vẫn còn bỏ sót một phần ca khó. Đây là điểm nên thảo luận ở failure analysis, đặc biệt với minority classes.

### 3. Kết quả test final nên báo cáo

Test final chạy một lần trên checkpoint `best.pt`, split `test`, image size 640, batch 16, seed 0, deterministic=True, Ultralytics 8.4.13, PyTorch 2.6.0+cu124, CUDA 12.4.

Kết quả test final:

| Split | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Test final | 0.8767 | 0.8202 | 0.9060 | 0.7247 | 0.8696 | 0.8134 | 0.9006 | 0.7154 |

Cách diễn giải mạnh vừa đủ:

- Mask mAP50 đạt 0.9006, cho thấy mô hình nhận diện vùng tổn thương tốt ở ngưỡng IoU 0.5.
- Mask mAP50-95 đạt 0.7154, cho thấy chất lượng mask vẫn giữ tốt khi yêu cầu IoU tăng từ 0.50 đến 0.95.
- Khoảng cách giữa box mAP50-95 0.7247 và mask mAP50-95 0.7154 chỉ khoảng 0.0093, cho thấy segmentation head không tụt nhiều so với localization.
- Mask precision 0.8696 cao hơn mask recall 0.8134, nên narrative nên nói mô hình có xu hướng conservative: ít false positive hơn, nhưng vẫn còn dư địa cải thiện recall.

Không nên diễn giải quá mức:

- Không nói test cao hơn validation nghĩa là model chắc chắn generalize tốt hơn mọi baseline. Có thể do split/test distribution, checkpoint selection và protocol khác.
- Không tune model/hyperparameter sau khi nhìn test final nếu vẫn dùng cùng test set làm kết quả cuối.
- Nếu sau này train v2A/v2B/v2C, phải so sánh bằng validation hoặc một test khác chưa mở; không dùng số test final này để chọn biến thể.

### 4. So sánh với RIVF26 YOLO26 baseline

PDF tham chiếu:

```text
D:\PAPER_SPKT\paper_base_ham1000\RIVF26__NCKH_K28_VietVu_DetaiSV_20260813_Submit.pdf
```

Thông tin chính từ paper baseline:

- Bài đó dùng fixed YOLO26n-seg để phân tích preprocessing/augmentation trên HAM10000.
- Protocol lesion-disjoint, test theo ba seed.
- Raw + online augmentation là cấu hình tốt nhất trong các input strategy chính.
- Kết quả YOLO26n-seg raw + online augmentation: Mask mAP50 = 0.7364 ± 0.0232, Mask mAP50:95 = 0.5636 ± 0.0234.
- Paper cũng nhấn mạnh lỗi class khó: mask overlap có thể cao nhưng diagnostic class vẫn sai; ví dụ các nhầm lẫn liên quan `mel/nv` và `nv/bkl`, và recall `akiec` thấp.

So sánh định hướng với SkinSeg-YOLO26-P2Attn:

| Method/source | Protocol note | Mask mAP50 | Mask mAP50-95 |
|---|---|---:|---:|
| RIVF26 YOLO26n-seg raw + online aug | lesion-disjoint, 3 seeds | 0.7364 ± 0.0232 | 0.5636 ± 0.0234 |
| SkinSeg-YOLO26-P2Attn final | current project split, 1 checkpoint/test final | 0.9006 | 0.7154 |

Câu nên viết:

> Compared with an earlier YOLO26n-seg preprocessing baseline reported under a lesion-disjoint three-seed protocol, the proposed SkinSeg-YOLO26-P2Attn obtains substantially higher mask mAP50 and mAP50-95 in the current experimental setting. Because the data split, training set construction, augmentation volume, and number of seeds are not identical, this comparison is used as contextual evidence rather than a controlled superiority claim.

Không nên viết:

> Our method outperforms the RIVF26 baseline by X% and proves SOTA.

Lý do không nên claim trực tiếp:

- Baseline PDF là three-seed lesion-disjoint protocol; run hiện tại là một checkpoint/test final trong project mới.
- Dữ liệu train hiện tại có multiview/offline augmentation lớn hơn.
- Cần thêm YOLO26 base cùng split/current pipeline để làm comparison chính thức. Anh sẽ bổ sung model YOLO26 base sau; lúc đó mới có đối chứng nội bộ thật sạch.

### 5. So sánh với YOLOv11 paper

PDF tham chiếu:

```text
D:\PAPER_SPKT\paper_base_ham1000\yolov11.pdf
```

Thông tin chính từ paper YOLOv11:

- Mô hình dùng YOLOv11s-seg trên HAM10000, có preprocessing/augmentation và deployment web app.
- Abstract/bảng kết quả báo detection mAP50 khoảng 0.91, detection mAP50-95 khoảng 0.735.
- Segmentation mAP50 khoảng 0.905, segmentation mAP50-95 khoảng 0.706.
- Phần kết luận báo gần tương tự: box mAP50 0.9145, box mAP50-95 0.7400, mask mAP50 0.9117, mask mAP50-95 0.7086.

So sánh định hướng:

| Method/source | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv11 paper | ~0.9145 | ~0.7400 | ~0.9117 | ~0.7086 |
| SkinSeg-YOLO26-P2Attn final | 0.9060 | 0.7247 | 0.9006 | 0.7154 |

Cách diễn giải:

- YOLOv11 paper có box metrics nhỉnh hơn một chút.
- SkinSeg-YOLO26-P2Attn có mask mAP50-95 nhỉnh hơn nhẹ so với con số segmentation mAP50-95 được báo cáo trong YOLOv11 paper.
- Khác biệt rất nhỏ và protocol không chắc đồng nhất, nên chỉ nên nói “comparable to” hoặc “competitive with”, không nói “outperforms” mạnh.
- Giá trị của paper YOLOv11 với bài mình nằm ở bối cảnh ứng dụng và mức metric tham chiếu, không phải là đối chứng ablation chính.

Câu nên viết:

> The proposed model achieves a mask mAP50-95 of 0.7154, which is competitive with a recent YOLOv11s-seg HAM10000 study reporting segmentation mAP50-95 around 0.706-0.7086. Since the reported protocols and splits are not guaranteed to be identical, the comparison is interpreted as contextual benchmarking.

### 6. Cách viết Results cho hội nghị

Có thể viết theo flow sau:

1. Nêu protocol: checkpoint selected on validation, final test evaluated once.
2. Báo validation best epoch 183 để chứng minh model selection không dùng test.
3. Báo test final table gồm box/mask P/R/mAP50/mAP50-95.
4. Nhấn mạnh mask mAP50 0.9006 và mask mAP50-95 0.7154.
5. Nói precision > recall, nên còn dư địa cải thiện recall/class khó.
6. So sánh contextual với RIVF26 baseline và YOLOv11 paper, nhưng gắn caveat protocol.
7. Kết thúc bằng limitation: cần YOLO26 base cùng split, ablation P2/CBAM/multiview/augmentation, nhiều seed và failure analysis theo class.

### 7. Cách viết Discussion

Điểm mạnh:

- P2 stride-4 path có cơ sở hợp lý cho boundary và lesion nhỏ.
- CBAM ở P2/P3 giúp refine feature sớm trước khi đi sâu hơn.
- Multi-view preprocessing giữ raw/letterbox view đồng thời thêm artifact-processed view, không thay val/test bằng ảnh xử lý mạnh.
- NV-excluding augmentation giảm áp lực class majority mà không tạo thêm NV vốn đã nhiều.

Điểm yếu cần nói thật:

- Test final hiện là một checkpoint, một seed.
- Chưa có YOLO26 base cùng exact current split để claim improvement nội bộ.
- Patient-level leakage chưa thể chứng minh nếu thiếu metadata bệnh nhân đầy đủ.
- Per-class confusion/failure chưa được phân tích sâu trong section này; cần bổ sung khi có confusion/classwise table.
- Recall thấp hơn precision, nên model vẫn có thể bỏ sót một số lesion hoặc ca minority khó.

### 8. Câu claim an toàn cho abstract hiện tại

Câu có thể dùng:

> On the held-out test split, SkinSeg-YOLO26-P2Attn achieves 0.9006 mask mAP50 and 0.7154 mask mAP50-95, with mask precision and recall of 0.8696 and 0.8134, respectively.

Câu có thể dùng nếu muốn so sánh nhẹ:

> These results are competitive with recent YOLO-based HAM10000 segmentation reports and exceed the earlier YOLO26n-seg preprocessing baseline in the current experimental setting, while controlled same-split ablations remain necessary for final attribution.

Câu không nên dùng:

> SkinSeg-YOLO26-P2Attn is state-of-the-art on HAM10000.

> The proposed preprocessing alone improves YOLO26 by 0.1518 mAP50-95.

> The model solves class imbalance.

### 9. Vị trí của v2A/v2B/v2C

V2A/v2B/v2C không nên đưa vào main claim hội nghị hiện tại. Nên đặt vào future work hoặc Q1 extension:

- v2A: ASPP-lite/DilatedContext để tăng deep context.
- v2B: GatedFusion ở P2/P3 để học adaptive fusion.
- v2C: kết hợp ASPP-lite và GatedFusion.

Câu future work:

> Future work will investigate context-aware and adaptive fusion variants, including ASPP-lite deep context and gated P2/P3 fusion, under the same validation-only model-selection protocol.

Kết luận section: bản hội nghị nên khóa ở SkinSeg-YOLO26-P2Attn với kết quả `SkinSeg_YOLO26_P2_CBAM_640_final`. Các biến thể v2 dùng cho Q1 để có câu chuyện architecture sâu hơn và ablation đầy đủ hơn.
