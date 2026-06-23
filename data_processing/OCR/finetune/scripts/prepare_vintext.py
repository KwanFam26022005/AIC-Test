#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chuẩn bị dataset VinText cho training PaddleOCR.
1. Download vietnamese_original.zip từ Google Drive bằng gdown.
2. Giải nén dataset.
3. Chia dataset theo đúng tỷ lệ:
   - Train (Ảnh 1-1200)
   - Val & Test (Ảnh 1201-1500) (Tránh tập Test bị rỗng do zip gốc chứa 1500 ảnh)
4. Convert annotation sang format SimpleDataSet của PaddleOCR:
   path/to/img.jpg\t[{"transcription": "text", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]
"""

import os
import sys
import shutil
import zipfile
import json
import argparse

# Khởi tạo argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Download và chuẩn bị dataset VinText cho PaddleOCR")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/OCR_finetune/data/vintext",
                        help="Thư mục output lưu dataset đã xử lý")
    parser.add_argument("--download_dir", type=str, default="./tmp_vintext",
                        help="Thư mục tạm để download và giải nén")
    parser.add_argument("--force_download", action="store_true", help="Bắt buộc download lại nếu file đã tồn tại")
    return parser.parse_args()

def download_and_extract(download_dir, force_download=False):
    """Download VinText original zip từ Google Drive và giải nén."""
    os.makedirs(download_dir, exist_ok=True)
    zip_path = os.path.join(download_dir, "vietnamese_original.zip")
    extract_path = os.path.join(download_dir, "extracted")
    
    # VinText Google Drive ID từ SEACrowd loader
    gdrive_id = "1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml"
    
    if not os.path.exists(zip_path) or force_download:
        print(f"[*] Đang tải xuống VinText từ Google Drive (ID: {gdrive_id})...")
        try:
            import gdown
        except ImportError:
            print("[!] Lỗi: Hãy cài đặt gdown bằng cách chạy: pip install gdown")
            sys.exit(1)
        
        gdown.download(id=gdrive_id, output=zip_path, quiet=False)
    else:
        print(f"[*] Đã tìm thấy zip file tại {zip_path}, bỏ qua download.")
        
    if not os.path.exists(extract_path) or force_download:
        print(f"[*] Đang giải nén {zip_path} vào {extract_path}...")
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("[*] Giải nén hoàn tất.")
    else:
        print(f"[*] Đã tìm thấy thư mục giải nén tại {extract_path}.")
        
    return extract_path

def convert_and_split(extract_path, output_dir):
    """Đọc ảnh và labels từ thư mục giải nén, chia split và convert format."""
    print("[*] Bắt đầu xử lý và chia split cho VinText...")
    
    # Cấu trúc giải nén thường là:
    # vietnamese/train_images -> im0001.jpg - im1500.jpg
    # vietnamese/test_image -> im1501.jpg - im2000.jpg
    # vietnamese/labels -> gt_1.txt - gt_2000.txt
    
    vietnamese_dir = os.path.join(extract_path, "vietnamese")
    if not os.path.exists(vietnamese_dir):
        # Trường hợp giải nén ra thẳng mà không có folder con
        # Thử tìm kiếm đệ quy
        subdirs = [os.path.join(extract_path, d) for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
        if subdirs:
            vietnamese_dir = subdirs[0]
            
    train_src_dir = os.path.join(vietnamese_dir, "train_images")
    test_src_dir = os.path.join(vietnamese_dir, "test_image")
    labels_src_dir = os.path.join(vietnamese_dir, "labels")
    
    # Kiểm tra tồn tại thư mục nguồn
    for d in [train_src_dir, test_src_dir, labels_src_dir]:
        if not os.path.exists(d):
            print(f"[!] Lỗi: Không tìm thấy thư mục nguồn: {d}")
            sys.exit(1)
            
    # Tạo cấu trúc thư mục đích
    train_dest_dir = os.path.join(output_dir, "train")
    val_dest_dir = os.path.join(output_dir, "val")
    test_dest_dir = os.path.join(output_dir, "test")
    
    for d in [train_dest_dir, val_dest_dir, test_dest_dir]:
        os.makedirs(d, exist_ok=True)
        
    # Chuẩn bị files label đích
    train_labels = []
    val_labels = []
    test_labels = []
    
    # Gom tất cả các ảnh từ cả 2 thư mục train_images và test_image
    all_images = []
    for filename in os.listdir(train_src_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append((os.path.join(train_src_dir, filename), filename))
            
    for filename in os.listdir(test_src_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append((os.path.join(test_src_dir, filename), filename))
            
    print(f"[*] Tìm thấy tổng cộng {len(all_images)} file ảnh.")
    
    processed_count = 0
    for src_img_path, img_name in all_images:
        # Lấy ID từ tên ảnh (ví dụ: im1234.jpg -> 1234)
        # Tên ảnh có thể có format "im0001.jpg" hoặc "im1.jpg"
        try:
            # Loại bỏ phần mở rộng và chữ cái đầu (vd: im)
            name_without_ext = os.path.splitext(img_name)[0]
            # Tìm số trong tên
            numeric_part = ''.join(c for c in name_without_ext if c.isdigit())
            img_id = int(numeric_part)
        except Exception as e:
            print(f"[!] Cảnh báo: Không thể parse ID từ tên ảnh: {img_name}. Bỏ qua file này. Lỗi: {e}")
            continue
            
        # Tìm label tương ứng: gt_{img_id}.txt
        gt_filename = f"gt_{img_id}.txt"
        gt_filepath = os.path.join(labels_src_dir, gt_filename)
        
        if not os.path.exists(gt_filepath):
            print(f"[!] Cảnh báo: Không tìm thấy file nhãn {gt_filepath} cho ảnh {img_name}. Bỏ qua.")
            continue
            
        # Đọc nhãn và parse thành PaddleOCR format
        annotations = []
        with open(gt_filepath, 'r', encoding='utf-8-sig') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Split tối đa 8 lần để tránh việc dấu phẩy trong transcript làm lỗi split
                parts = line.split(',', 8)
                if len(parts) < 9:
                    # Đôi khi có dòng bị trống hoặc lỗi format
                    print(f"[!] Cảnh báo: Dòng {line_no} trong {gt_filename} không đủ cột: {line}")
                    continue
                    
                try:
                    # Tọa độ 4 điểm: x1,y1,x2,y2,x3,y3,x4,y4
                    coords = [int(float(p.strip())) for p in parts[:8]]
                    points = [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]],
                        [coords[4], coords[5]],
                        [coords[6], coords[7]]
                    ]
                    transcript = parts[8].strip()
                    
                    annotations.append({
                        "transcription": transcript,
                        "points": points
                    })
                except Exception as e:
                    print(f"[!] Lỗi parse dòng {line_no} trong {gt_filename}: {e}")
                    
        # Xác định split và copy ảnh
        if img_id <= 1200:
            dest_img_path = os.path.join(train_dest_dir, img_name)
            shutil.copy2(src_img_path, dest_img_path)
            relative_path = f"train/{img_name}"
            train_labels.append(f"{relative_path}\t{json.dumps(annotations, ensure_ascii=False)}")
        else:
            # Lưu đồng thời vào val và test splits
            val_img_path = os.path.join(val_dest_dir, img_name)
            shutil.copy2(src_img_path, val_img_path)
            val_labels.append(f"val/{img_name}\t{json.dumps(annotations, ensure_ascii=False)}")
            
            test_img_path = os.path.join(test_dest_dir, img_name)
            shutil.copy2(src_img_path, test_img_path)
            test_labels.append(f"test/{img_name}\t{json.dumps(annotations, ensure_ascii=False)}")
            
        processed_count += 1
        
    # Ghi file label.txt ra Drive
    write_label_file(os.path.join(output_dir, "train_label.txt"), train_labels)
    write_label_file(os.path.join(output_dir, "val_label.txt"), val_labels)
    write_label_file(os.path.join(output_dir, "test_label.txt"), test_labels)
    
    print(f"\n[+] Xử lý hoàn tất {processed_count} ảnh.")
    print(f"    - Train: {len(train_labels)} ảnh -> {os.path.join(output_dir, 'train_label.txt')}")
    print(f"    - Val: {len(val_labels)} ảnh -> {os.path.join(output_dir, 'val_label.txt')}")
    print(f"    - Test: {len(test_labels)} ảnh -> {os.path.join(output_dir, 'test_label.txt')}")

def write_label_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

def main():
    args = parse_args()
    
    print("=== PADDLEOCR VINTEXT DATASET PREPARATION ===")
    print(f"[*] Output directory: {args.output_dir}")
    print(f"[*] Temporary download directory: {args.download_dir}")
    
    # 1. Download & Extract
    extract_path = download_and_extract(args.download_dir, args.force_download)
    
    # 2. Convert and split
    convert_and_split(extract_path, args.output_dir)
    
    # 3. Dọn dẹp thư mục tạm (tùy chọn, ở đây giữ lại zip để tránh tải lại)
    # shutil.rmtree(extract_path)
    print("[+] Hoàn thành toàn bộ quy trình chuẩn bị dữ liệu!")

if __name__ == "__main__":
    main()
