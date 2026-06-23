#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script đánh giá (evaluation) model PP-OCRv6_medium_det trên VinText test set.
1. Nhận đường dẫn checkpoint và file config tương ứng.
2. Thiết lập cấu hình đánh giá để trỏ tới test split của VinText.
3. Đăng ký custom transforms (MotionBlur, GaussianNoise) để tránh lỗi load config.
4. Chạy evaluation và bắt log đầu ra để thu thập metrics (precision, recall, hmean).
5. Ghi nhận kết quả vào file CSV (logs/eval_history.csv).
"""

import os
import sys
import json
import logging
import argparse
import time
import csv
from datetime import datetime

# Định nghĩa các thư mục mặc định trên Google Drive
DRIVE_DIR = "/content/drive/MyDrive/OCR_finetune"
EVAL_HISTORY_FILE = os.path.join(DRIVE_DIR, "logs/eval_history.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá model PP-OCRv6 Det trên Test Set")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn tới file config yml (vd: finetune_phase2.yml)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn tới file checkpoint .pdparams cần đánh giá (không bao gồm phần mở rộng)")
    parser.add_argument("--drive_dir", type=str, default=DRIVE_DIR, help="Đường dẫn gốc Google Drive")
    parser.add_argument("--phase", type=int, default=0, help="Phase hiện tại (dùng để log vào csv)")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch hiện tại (dùng để log vào csv)")
    return parser.parse_args()

class MetricsHandler(logging.Handler):
    """Handler thu thập log in ra từ PaddleOCR để trích xuất metrics."""
    def __init__(self):
        super().__init__()
        self.metrics = None

    def emit(self, record):
        log_msg = record.getMessage()
        if 'metric eval_info' in log_msg:
            try:
                # Trích xuất string dictionary sau "metric eval_info:"
                dict_str = log_msg.split('metric eval_info:', 1)[1].strip()
                import ast
                self.metrics = ast.literal_eval(dict_str)
                print(f"[+] Trích xuất metrics thành công: {self.metrics}")
            except Exception as e:
                print(f"[!] Lỗi khi parse metrics từ log: {e}")

def log_metrics_to_csv(csv_path, phase, epoch, checkpoint_name, metrics):
    """Ghi nhận thông tin evaluation vào file CSV lịch sử."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": phase,
        "epoch": epoch,
        "checkpoint": os.path.basename(checkpoint_name),
        "precision": round(metrics.get("precision", 0.0), 4),
        "recall": round(metrics.get("recall", 0.0), 4),
        "hmean": round(metrics.get("hmean", 0.0), 4)
    }
    
    fieldnames = ["timestamp", "phase", "epoch", "checkpoint", "precision", "recall", "hmean"]
    
    try:
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[+] Kết quả eval đã được lưu vào {csv_path}")
    except Exception as e:
        print(f"[!] Không thể ghi file CSV kết quả: {e}")

def main():
    # Đảm bảo thư mục hiện tại nằm trong python path để import được ppocr
    if os.path.exists('./tools/eval.py'):
        sys.path.insert(0, os.path.abspath('.'))
        
    args = parse_args()
    
    print("=== PADDLEOCR DETECTION TEST EVALUATION ===")
    print(f"[*] Config file: {args.config}")
    print(f"[*] Checkpoint: {args.checkpoint}")
    print(f"[*] Test Label: {os.path.join(args.drive_dir, 'data/vintext/test_label.txt')}")
    
    # 1. Đăng ký custom transforms (MotionBlur, GaussianNoise)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    try:
        from custom_transforms import register_custom_transforms
        register_custom_transforms()
    except ImportError as e:
        print(f"[!] Cảnh báo: Không tìm thấy custom_transforms.py: {e}")
        
    # 2. Cài đặt logging handler để bắt metrics
    ppocr_logger = logging.getLogger('ppocr')
    metrics_handler = MetricsHandler()
    ppocr_logger.addHandler(metrics_handler)
    
    # 3. Chuẩn bị sys.argv để chạy tools/eval.py của PaddleOCR
    # Thiết lập ghi đè để bắt buộc đánh giá trên TEST split của VinText
    test_data_dir = os.path.join(args.drive_dir, "data/vintext/")
    test_label_list = [os.path.join(args.drive_dir, "data/vintext/test_label.txt")]
    
    sys.argv = [
        'tools/eval.py',
        '-c', args.config,
        '-o', f'Global.pretrained_model={args.checkpoint}',
        f'Global.checkpoints=null',
        f'Eval.dataset.data_dir={test_data_dir}',
        f'Eval.dataset.label_file_list={test_label_list}',
        f'Eval.loader.batch_size_per_card=1', # Eval luôn dùng batch size 1
        f'Eval.loader.num_workers=2'
    ]
    
    print(f"[*] Khởi chạy PaddleOCR eval với argv: {sys.argv}")
    
    # Đảm bảo PaddleOCR nằm trong python path (Đã thực hiện ở đầu hàm main)
    if not os.path.exists('./tools/eval.py'):
        print("[!] Cảnh báo: Không tìm thấy tools/eval.py ở thư mục hiện tại.")
        
    # 4. Import tools.eval và chạy main (Kiểm tra tương thích signature main)
    try:
        import tools.eval as paddle_eval
        import tools.program as program
        import inspect
        
        # Chạy evaluation chính
        sig = inspect.signature(paddle_eval.main)
        if len(sig.parameters) == 0:
            paddle_eval.main()
        else:
            config, device, logger, vdl_writer = program.preprocess()
            paddle_eval.main(config, device, logger, vdl_writer)
        
        # 5. Ghi nhận kết quả
        if metrics_handler.metrics:
            log_metrics_to_csv(
                csv_path=EVAL_HISTORY_FILE,
                phase=args.phase,
                epoch=args.epoch,
                checkpoint_name=args.checkpoint,
                metrics=metrics_handler.metrics
            )
            print("\n" + "="*50)
            print(f"[+] KẾT QUẢ ĐÁNH GIÁ TRÊN TEST SET:")
            print(f"    - Precision: {metrics_handler.metrics.get('precision', 0.0):.4f}")
            print(f"    - Recall:    {metrics_handler.metrics.get('recall', 0.0):.4f}")
            print(f"    - Hmean:     {metrics_handler.metrics.get('hmean', 0.0):.4f}")
            print("="*50 + "\n")
        else:
            print("[!] Cảnh báo: Không tìm thấy thông tin metrics từ log của PaddleOCR.")
            
    except Exception as e:
        print(f"[!] Lỗi khi chạy evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
