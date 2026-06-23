#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Master script chạy fine-tuning PP-OCRv6_medium_det.
1. Tự động detect GPU (T4/L4/A100) để thiết lập batch size tối ưu.
2. Quản lý trạng thái training qua file JSON trên Drive (training_progress.json).
3. Hỗ trợ tự động resume từ checkpoint mới nhất nếu bị ngắt kết nối.
4. Đăng ký custom transforms (MotionBlur, GaussianNoise).
5. Freeze Backbone ở Phase 1 bằng cách set stop_gradient = True cho backbone params.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import threading
import re

# Định nghĩa các thư mục mặc định trên Google Drive
DRIVE_DIR = "/content/drive/MyDrive/OCR_finetune"
PROGRESS_FILE = os.path.join(DRIVE_DIR, "logs/training_progress.json")
EVAL_HISTORY_FILE = os.path.join(DRIVE_DIR, "logs/eval_history.csv")

def parse_args():
    parser = argparse.ArgumentParser(description="Master Fine-tuning Runner cho PaddleOCR Det")
    parser.add_argument("--drive_dir", type=str, default=DRIVE_DIR, help="Đường dẫn gốc Google Drive")
    parser.add_argument("--phase", type=int, default=None, help="Chỉ định chạy cứng phase (1, 2, 3). Nếu để None sẽ chạy tự động theo progress.")
    parser.add_argument("--local_config_dir", type=str, default="./configs", help="Thư mục chứa config file cục bộ")
    return parser.parse_args()

def detect_gpu():
    """Detect GPU type và trả về cấu hình tối ưu."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader'], capture_output=True, text=True)
        gpu_info = result.stdout.strip()
        print(f"[*] Kết quả detect GPU: {gpu_info}")
        
        if 'A100' in gpu_info:
            return {
                'gpu': 'A100',
                'vram_gb': 40,
                'batch_size': {1: 32, 2: 16, 3: 8},
                'num_workers': 4
            }
        elif 'L4' in gpu_info:
            return {
                'gpu': 'L4',
                'vram_gb': 24,
                'batch_size': {1: 16, 2: 8, 3: 4},
                'num_workers': 4
            }
        else: # Mặc định coi là T4
            return {
                'gpu': 'T4',
                'vram_gb': 16,
                'batch_size': {1: 8, 2: 4, 3: 2},
                'num_workers': 2
            }
    except Exception as e:
        print(f"[!] Lỗi khi chạy nvidia-smi ({e}), sử dụng cấu hình mặc định cho CPU/T4.")
        return {
            'gpu': 'Default/T4',
            'vram_gb': 16,
            'batch_size': {1: 8, 2: 4, 3: 2},
            'num_workers': 2
        }

def load_progress(progress_file):
    """Đọc file tiến trình training_progress.json."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Lỗi đọc file tiến trình: {e}. Tạo mới.")
            
    # Khởi tạo tiến trình mặc định
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    initial_progress = {
        "current_phase": 1,
        "phases": {
            "1": {
                "status": "pending",
                "total_epochs": 40,
                "completed_epochs": 0,
                "best_hmean": 0.0,
                "best_checkpoint": "",
                "gpu_type": ""
            },
            "2": {
                "status": "pending",
                "total_epochs": 60,
                "completed_epochs": 0,
                "best_hmean": 0.0,
                "best_checkpoint": "",
                "gpu_type": ""
            },
            "3": {
                "status": "pending",
                "total_epochs": 30,
                "completed_epochs": 0,
                "best_hmean": 0.0,
                "best_checkpoint": "",
                "gpu_type": ""
            }
        },
        "eval_history": []
    }
    save_progress(progress_file, initial_progress)
    return initial_progress

def save_progress(progress_file, progress_data):
    """Lưu file tiến trình training_progress.json."""
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Không thể lưu file tiến trình: {e}")

def update_progress_epoch(progress_file, phase_key, epoch):
    """Cập nhật số epoch đã hoàn thành của phase hiện tại."""
    if os.path.exists(progress_file):
        try:
            # Đọc nhanh để tránh ghi đè dữ liệu mới
            with open(progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["phases"][str(phase_key)]["completed_epochs"] = epoch
            if data["phases"][str(phase_key)]["status"] == "pending":
                data["phases"][str(phase_key)]["status"] = "in_progress"
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

class LogMonitorThread(threading.Thread):
    """Thread đọc file log của PaddleOCR để cập nhật tiến trình epoch thời gian thực."""
    def __init__(self, log_filepath, progress_filepath, phase_key):
        super().__init__()
        self.log_filepath = log_filepath
        self.progress_filepath = progress_filepath
        self.phase_key = phase_key
        self.daemon = True
        self.stop_event = threading.Event()
        
    def run(self):
        print(f"[*] Thread monitor log đã khởi động cho: {self.log_filepath}")
        last_position = 0
        while not self.stop_event.is_set():
            if os.path.exists(self.log_filepath):
                try:
                    with open(self.log_filepath, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        lines = f.readlines()
                        last_position = f.tell()
                        
                        latest_epoch = None
                        for line in lines:
                            # PaddleOCR format logs: "epoch: [5/40], iter: 10..."
                            match = re.search(r'epoch:\s*\[(\d+)/\d+\]', line)
                            if match:
                                latest_epoch = int(match.group(1))
                                
                        if latest_epoch is not None:
                            update_progress_epoch(self.progress_filepath, self.phase_key, latest_epoch)
                except Exception as e:
                    pass
            time.sleep(15)
            
    def stop(self):
        self.stop_event.set()

def get_completed_epochs_from_states(checkpoint_dir):
    """Đọc file states để biết epoch hoàn thành thực tế."""
    states_path = os.path.join(checkpoint_dir, "latest.states")
    if os.path.exists(states_path):
        try:
            import paddle
            states = paddle.load(states_path)
            if 'epoch' in states:
                return states['epoch']
            elif 'epoch_id' in states:
                return states['epoch_id']
        except Exception as e:
            print(f"[*] Không thể đọc file .states: {e}")
    return 0

def setup_backbone_freezing(should_freeze):
    """Monkey-patch ppocr.modeling.architectures.build_model để freeze backbone nếu được yêu cầu."""
    try:
        import ppocr.modeling.architectures as architectures
        original_build_model = architectures.build_model
        
        def custom_build_model(config):
            model = original_build_model(config)
            if should_freeze:
                print("\n" + "="*50)
                print("[*] ĐANG THỰC HIỆN FREEZE BACKBONE (PPLCNetV4)...")
                print("="*50)
                freeze_count = 0
                for name, param in model.backbone.named_parameters():
                    param.stop_gradient = True
                    freeze_count += 1
                print(f"[+] Đã đóng băng {freeze_count} tham số của Backbone.")
                
                # Check số lượng tham số trainable còn lại
                trainable_count = 0
                for name, param in model.named_parameters():
                    if not param.stop_gradient:
                        trainable_count += 1
                print(f"[+] Số lượng tham số tiếp tục train (Neck + Head): {trainable_count}")
                print("="*50 + "\n")
            return model
            
        architectures.build_model = custom_build_model
        print("[+] Đăng ký monkeypatch freeze backbone thành công.")
    except Exception as e:
        print(f"[!] Lỗi khi thiết lập freeze backbone: {e}")

def main():
    # Đảm bảo thư mục hiện tại nằm trong python path để import được ppocr
    if os.path.exists('./tools/train.py'):
        sys.path.insert(0, os.path.abspath('.'))
        
    args = parse_args()
    
    # 1. Detect GPU
    gpu_config = detect_gpu()
    print(f"[+] Sử dụng GPU: {gpu_config['gpu']} | VRAM lý thuyết: {gpu_config['vram_gb']}GB")
    
    # 2. Đăng ký custom transforms (MotionBlur, GaussianNoise)
    # Tìm file custom_transforms.py trong cùng thư mục script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    try:
        from custom_transforms import register_custom_transforms
        register_custom_transforms()
    except ImportError as e:
        print(f"[!] Không thể import custom_transforms.py: {e}")
        print("[!] Đảm bảo custom_transforms.py nằm trong cùng thư mục với train_runner.py")
        sys.exit(1)
        
    # 3. Load Progress
    progress = load_progress(PROGRESS_FILE)
    
    # Xác định phase cần chạy
    if args.phase is not None:
        run_phase = args.phase
        print(f"[*] Người dùng chỉ định chạy cứng Phase {run_phase}")
    else:
        run_phase = progress["current_phase"]
        print(f"[*] Chạy tự động: Phase hiện tại là Phase {run_phase}")
        
    if run_phase > 3:
        print("[+] Tất cả 3 phase đã hoàn thành! Không cần chạy thêm.")
        sys.exit(0)
        
    phase_str = str(run_phase)
    phase_info = progress["phases"][phase_str]
    
    # Tìm file config tương ứng
    config_name = f"finetune_phase{run_phase}.yml"
    config_path = os.path.join(args.drive_dir, "configs", config_name)
    if not os.path.exists(config_path):
        # Thử tìm ở thư mục local
        config_path = os.path.join(args.local_config_dir, config_name)
        
    if not os.path.exists(config_path):
        print(f"[!] Lỗi: Không tìm thấy file cấu hình {config_name}")
        sys.exit(1)
        
    print(f"[+] Sử dụng file config: {config_path}")
    
    # Xác định checkpoint và resume status
    checkpoint_dir = os.path.join(args.drive_dir, f"checkpoints/phase{run_phase}_" + 
                                  ("freeze" if run_phase == 1 else "unfreeze" if run_phase == 2 else "highres"))
    
    # Kiểm tra xem checkpoint latest có tồn tại không
    latest_params_path = os.path.join(checkpoint_dir, "latest.pdparams")
    
    is_resume = False
    if os.path.exists(latest_params_path):
        is_resume = True
        completed_epochs = get_completed_epochs_from_states(checkpoint_dir)
        print(f"[*] Phát hiện checkpoint tồn tại. Sẽ RESUME từ Epoch {completed_epochs}...")
    else:
        print(f"[*] Không thấy checkpoint cũ cho Phase {run_phase}. Sẽ bắt đầu training mới.")
        
    # Chuẩn bị tham số dòng lệnh cho PaddleOCR
    batch_size = gpu_config['batch_size'][run_phase]
    num_workers = gpu_config['num_workers']
    
    # Thiết lập monkeypatch freeze backbone cho Phase 1
    setup_backbone_freezing(should_freeze=(run_phase == 1))
    
    # Build list command-line overrides (-o)
    overrides = []
    
    # Override batch size và workers phù hợp với GPU hiện tại
    overrides.append(f"Train.loader.batch_size_per_card={batch_size}")
    overrides.append(f"Train.loader.num_workers={num_workers}")
    overrides.append(f"Global.save_model_dir={checkpoint_dir}")
    
    # Cấu hình pretrained_model hoặc checkpoints (resume)
    if is_resume:
        # Khi resume: dùng checkpoints key, và pretrained_model phải set thành null
        overrides.append(f"Global.checkpoints={os.path.join(checkpoint_dir, 'latest')}")
        overrides.append("Global.pretrained_model=null")
    else:
        # Khi chạy mới: checkpoints = null
        overrides.append("Global.checkpoints=null")
        if run_phase == 1:
            # Phase 1 bắt đầu từ pretrained weights chính thức
            pretrained_weights = os.path.join(args.drive_dir, "pretrained/PP-OCRv6_medium_det")
            overrides.append(f"Global.pretrained_model={pretrained_weights}")
        elif run_phase == 2:
            # Phase 2 bắt đầu từ best của Phase 1
            prev_best = os.path.join(args.drive_dir, "checkpoints/phase1_freeze/best_accuracy")
            overrides.append(f"Global.pretrained_model={prev_best}")
        elif run_phase == 3:
            # Phase 3 bắt đầu từ best của Phase 2
            prev_best = os.path.join(args.drive_dir, "checkpoints/phase2_unfreeze/best_accuracy")
            overrides.append(f"Global.pretrained_model={prev_best}")
            
    # Ghi nhận thông tin chạy hiện tại vào progress
    progress["phases"][phase_str]["status"] = "in_progress"
    progress["phases"][phase_str]["gpu_type"] = gpu_config['gpu']
    save_progress(PROGRESS_FILE, progress)
    
    # Thiết lập sys.argv cho PaddleOCR tools/train.py
    sys.argv = ['tools/train.py', '-c', config_path]
    for ov in overrides:
        sys.argv.extend(['-o', ov])
        
    print(f"[*] Khởi chạy PaddleOCR train với argv: {sys.argv}")
    
    # Đảm bảo PaddleOCR nằm trong python path (Đã thực hiện ở đầu hàm main)
    if not os.path.exists('./tools/train.py'):
        print("[!] Cảnh báo: Không tìm thấy tools/train.py ở thư mục hiện tại. Đang thử chạy giả định import...")
        
    # Tạo thư mục checkpoint trước để tránh lỗi log file ghi không được
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "train.log")
    
    # Khởi động thread monitor log để update JSON thường xuyên
    monitor = LogMonitorThread(log_file, PROGRESS_FILE, run_phase)
    monitor.start()
    
    # Import tools.train và thực thi main
    try:
        import tools.train as paddle_train
        
        print("\n" + "="*50)
        print(f"[*] BẮT ĐẦU TRAINING PHASE {run_phase}...")
        print("="*50 + "\n")
        
        # Chạy train loop chính của PaddleOCR (Kiểm tra tương thích signature main)
        import inspect
        sig = inspect.signature(paddle_train.main)
        if len(sig.parameters) == 0:
            paddle_train.main()
        else:
            import tools.program as program
            config, device, logger, vdl_writer = program.preprocess()
            paddle_train.main(config, device, logger, vdl_writer)
        
        # Kết thúc thành công
        monitor.stop()
        monitor.join()
        
        print("\n" + "="*50)
        print(f"[+] PHASE {run_phase} TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print("="*50 + "\n")
        
        # Cập nhật progress JSON
        progress = load_progress(PROGRESS_FILE)
        progress["phases"][phase_str]["status"] = "completed"
        progress["phases"][phase_str]["completed_epochs"] = progress["phases"][phase_str]["total_epochs"]
        
        # Tìm best checkpoint path
        best_ckpt_path = os.path.join(checkpoint_dir, "best_accuracy")
        progress["phases"][phase_str]["best_checkpoint"] = best_ckpt_path
        
        # Chuyển sang phase tiếp theo
        next_phase = run_phase + 1
        progress["current_phase"] = next_phase
        
        save_progress(PROGRESS_FILE, progress)
        print(f"[*] Đã cập nhật tiến trình. Phase tiếp theo: {next_phase}")
        
    except KeyboardInterrupt:
        print("[!] Tiến trình bị hủy bởi người dùng.")
        monitor.stop()
        sys.exit(0)
    except Exception as e:
        print(f"[!] Lỗi nghiêm trọng xảy ra trong quá trình training: {e}")
        monitor.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
