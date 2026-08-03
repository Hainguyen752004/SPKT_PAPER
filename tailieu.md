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
2. Cung cấp tín hiệu có độ phân giải cao hơn cho mask prototype và mask coefficients của `Segment26`.
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
