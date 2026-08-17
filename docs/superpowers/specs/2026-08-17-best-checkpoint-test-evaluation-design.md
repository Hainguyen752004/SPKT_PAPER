# Best Checkpoint Test Evaluation Design

## Mục tiêu

Thêm một script độc lập để đánh giá checkpoint `best.pt` của mô hình SkinSeg-YOLO26n-P2-CBAM trên đúng split `test`, lưu toàn bộ artifact đánh giá vào một run mới và không thay đổi dữ liệu, checkpoint hoặc run train hiện có.

## Giao diện dòng lệnh

Script mới: `04_evaluate_best_test.py`.

- `--weights`: đường dẫn checkpoint; mặc định là `D:/PAPER_SPKT/SkinSeg_YOLO26_P2_CBAM_640/weights/best.pt` khi đường dẫn này tồn tại.
- `--data`: đường dẫn dataset YAML; mặc định `dataset_p2_cbam.yaml` cạnh script.
- `--batch`: batch size dương, mặc định 16.
- `--workers`: số worker không âm; mặc định 0 trên Windows và 8 trên Linux có CUDA.
- `--device`: tùy chọn ghi đè device; mặc định GPU 0 nếu CUDA khả dụng, ngược lại CPU.
- `--name`: tên run; mặc định `SkinSeg_YOLO26_P2_CBAM_Test`.
- `--seed`: seed đánh giá, mặc định 0.

Script cố định `split="test"`, `imgsz=640`, `plots=True`, `save_json=True`, `iou=0.7`, `deterministic=True` và không cung cấp tùy chọn đổi split để tránh vô tình báo validation thành test. Không truyền `conf=None`; để Ultralytics dùng default và ghi effective arguments trả về nếu API cung cấp.

Phiên bản thực thi chính thức là `ultralytics==8.4.13`, khớp `requirements.txt` và môi trường đã tạo checkpoint. Script kiểm tra `ultralytics.__version__` và fail rõ ràng nếu khác. Thông báo 8.4.60 trong script train/README là bất nhất tài liệu cần sửa riêng, không được âm thầm đổi môi trường khi đánh giá checkpoint này.

## Luồng xử lý

1. Đăng ký `CBAM` và `P2CompatibleSegment26` trước khi giải tuần tự checkpoint.
2. Resolve đường dẫn tuyệt đối, kiểm tra checkpoint tồn tại/có đuôi `.pt`, YAML tồn tại và chỉ chấp nhận `test` dạng một đường dẫn thư mục. Xác nhận `images/test` cùng `labels/test` tồn tại mà không yêu cầu train/val. Không cần tạo YAML tạm.
3. Chỉ chấp nhận `--name` là một tên thư mục đơn, không có separator; resolve `project = SCRIPT_DIR/runs/segment` và xác nhận `save_dir = project/name` vẫn nằm bên trong project. Nếu `save_dir` đã tồn tại thì dừng trước khi khởi tạo `YOLO`. Giữ sibling lock `.name.evaluate.lock` bằng exclusive-create trong toàn bộ lần chạy và xóa lock trong `finally`; lock có sẵn thì dừng. Sau `val`, bắt buộc `Path(metrics.save_dir).resolve() == save_dir`; khác biệt (ví dụ Ultralytics tạo `name2`) là lỗi, không được viết summary nhầm vị trí.
4. Load checkpoint bằng `YOLO(weights)`, sau đó xác nhận task là segmentation và head là custom `P2CompatibleSegment26` với bốn mức stride `[4, 8, 16, 32]`.
5. Gọi `model.val(...)` đúng một lần với `split="test"`.
6. Lưu artifact Ultralytics trong absolute `project` và `<name>`, vẫn truyền `exist_ok=False`; preflight ở bước 3 mới là cơ chế chống ghi đè chính.
7. Trích `metrics.results_dict` thành `test_metrics.json` tại chính `Path(metrics.save_dir)`. Contract bắt buộc gồm tám key của Ultralytics 8.4.13: `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `metrics/precision(M)`, `metrics/recall(M)`, `metrics/mAP50(M)`, `metrics/mAP50-95(M)`. Mọi Tensor/NumPy scalar phải được đổi thành số Python hữu hạn; key vắng mặt được ghi `null` và đưa vào `missing_metrics`.
8. JSON có schema version, metric dictionary, checkpoint/data tuyệt đối, SHA-256 checkpoint, split, image size, batch, workers, requested/resolved device, seed, save dir, Python/Ultralytics/PyTorch/CUDA versions và timestamp UTC. `requested_val_args` luôn là dictionary đầy đủ được script tạo và truyền vào `val`; `effective_val_args` lấy từ validator nếu khả dụng, nếu không dùng bản sao `requested_val_args` cộng `conf: 0.001` là default validation đã khóa cho phiên bản 8.4.13 và đánh dấu `effective_args_source: "pinned_fallback"`. Ghi file atomically bằng temp file cùng thư mục rồi `os.replace`; lỗi không được ghi đè summary cũ hay che exception gốc.

## An toàn và lỗi

- Thiếu checkpoint, YAML, test images hoặc test labels: dừng trước khi gọi Ultralytics và báo đường dẫn cụ thể.
- Thư mục output đã tồn tại: dừng trước khi load checkpoint; người dùng đổi `--name`.
- Không xóa hay sửa runtime artifact.
- Không gọi `train`, không resume và không điều chỉnh threshold theo kết quả test.
- Đây là lần mở test cuối duy nhất cho checkpoint đã khóa. Trước khi chạy, `tailieu.md` phải chứa ma trận ablation A–D và quy tắc chọn checkpoint bằng validation. Metric test này tuyệt đối không được dùng để chọn A–D, attention placement, augmentation hoặc hyperparameter. Nếu anh muốn tiếp tục chọn model dựa trên kết quả nhìn thấy, phải giữ một final test set khác chưa từng mở; nếu không có, mọi phát triển tiếp theo chỉ dùng validation và kết quả này chỉ là báo cáo checkpoint hiện tại.

## Kiểm thử

Mở rộng `tests/test_model_architecture.py` hoặc tạo `tests/test_evaluate_best_test.py` để kiểm tra:

- default checkpoint/data được resolve đúng;
- input số học không hợp lệ bị từ chối;
- thiếu thư mục test bị báo trước khi evaluate;
- lời gọi giả lập tới `YOLO.val` luôn chứa `split="test"`, `imgsz=640`, `plots=True` và không ghi đè;
- JSON summary chứa schema/provenance/checksum/version, finite metric hoặc `null`, và được ghi atomically vào đúng `metrics.save_dir`;
- không có lời gọi train.
- thiếu input, output collision và tên thoát project đều fail trước khi khởi tạo `YOLO`;
- đăng ký custom module xảy ra trước load checkpoint, head/task/stride sai bị từ chối;
- `val()` được gọi đúng một lần; mọi lỗi không để lại summary một phần.
- version khác 8.4.13 bị từ chối; tám metric key có contract ổn định; fallback effective args hoạt động;
- lock collision và returned `metrics.save_dir` khác expected đều bị từ chối, lock luôn được cleanup trong `finally`.

Do máy Windows hiện tại không khởi chạy được `python.exe`, kiểm tra tĩnh vẫn được chạy cục bộ; test pytest đầy đủ cần chạy trong môi trường Python/Ultralytics dùng để train.

## Cập nhật tài liệu

Nối đúng một lần vào cuối `tailieu.md` một mục ngày 2026-08-17 và giữ nguyên UTF-8/nội dung cũ. Nội dung bắt buộc gồm:

- CBAM channel/spatial đúng công thức, giữ shape; P2–P5 có stride 4/8/16/32; prototype dùng P3–P5 và ở stride 4; run 200 epoch chứng minh graph/loss hoạt động.
- YAML đúng cho scale `n` nhưng channel CBAM 64/128 đang ghi cứng, không tự an toàn cho s/m/l/x.
- Validation best khoảng epoch 183: mask mAP50 0.8664, mask mAP50–95 0.6874, box mAP50–95 0.7118; epoch 200 thấp hơn 0.0017 mask mAP50–95; val segmentation loss tăng nhẹ cuối run nên dùng `best.pt`.
- Confusion: `nv/bcc/df/vasc` mạnh hơn; `mel/akiec` yếu; rủi ro `mel→nv`, `akiec→bkl/background`.
- Class polygon: nv 24138, mel 9336, bkl 9220, bcc 4316, akiec 2742, vasc 1188, df 966; NV-excluding augmentation chưa cân bằng các minority class với nhau.
- `optimizer=auto`, rotation 180°, mixup/copy-paste và augmentation online+offline là biến gây nhiễu cần ablate; đề xuất đối chứng AdamW/lr 0.001 và augmentation ít gây ảnh da không thực hơn.
- Baseline PDF test ba seed 0.5636±0.0234 mask mAP50–95 không được so trực tiếp với validation một seed 0.6874 trên 31.880 train images.
- Ma trận: A=YOLO26n baseline không P2/CBAM; B=P2-only; C=CBAM-only; D=P2+CBAM, tất cả cùng split/data/seed/hyperparameter/checkpoint rule.
- Quy tắc: model/hyperparameter chọn bằng validation; test chỉ dùng báo cáo cuối, không dùng để chỉnh model.
