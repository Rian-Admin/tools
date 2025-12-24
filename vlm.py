#!/usr/bin/env python3
"""
하이브리드 VLM 분석 및 파일 정리 통합 시스템
- 전체 이미지 분석 + 크롭 영역 분석 수행
- JSON 결과 저장 + 판정별 파일 자동 분리
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import os
import re
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter

try:
    import torch
    from PIL import Image
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 의존성 라이브러리가 없습니다: {e}")
    print("pip install torch torchvision transformers pillow")
    DEPENDENCIES_AVAILABLE = False

# 프롬프트들
PROMPTS = [
    "USER: <image>\nCarefully examine this image. Do you see any birds, ducks, geese, or other waterfowl? "
    "Answer with JSON format: {\"label\":\"bird\",\"reason\":\"I can see a duck swimming\"} "
    "or {\"label\":\"background\",\"reason\":\"only water and vegetation\"}\nASSISTANT:",
    
    "USER: <image>\nLook closely at this image. Are there any living birds visible? "
    "This includes ducks, geese, swans, or any other waterfowl. "
    "Respond only in JSON: {\"label\":\"bird\",\"reason\":\"description of what you see\"} "
    "or {\"label\":\"background\",\"reason\":\"no birds visible\"}\nASSISTANT:",
    
    "USER: <image>\nIs there a bird in this image? "
    "Answer as JSON: {\"label\":\"bird\",\"reason\":\"bird description\"} "
    "or {\"label\":\"background\",\"reason\":\"no bird\"}\nASSISTANT:"
]

JSON_PAT = re.compile(r'\{[^}]*"label"[^}]*\}', re.S)

def load_model(model_name="llava-hf/llava-1.5-7b-hf", precision="fp16"):
    """모델 로드"""
    if not DEPENDENCIES_AVAILABLE:
        return None, None
    
    torch_dtype = {
        "bf16": torch.bfloat16, 
        "fp16": torch.float16, 
        "fp32": torch.float32
    }[precision]
    
    print(f"모델 로드 중: {model_name}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    return model, processor

def run_vlm_inference(image: Image.Image, model, processor, use_multi_prompt=True) -> Dict[str, Any]:
    """VLM 추론 실행"""
    if not DEPENDENCIES_AVAILABLE or model is None:
        return {"label": "background", "reason": "dependencies-not-available", "raw": "", "prompt_id": -1}
    
    prompts = PROMPTS if use_multi_prompt else [PROMPTS[0]]
    results = []
    
    for i, prompt in enumerate(prompts):
        try:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=64,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            text = processor.batch_decode(out, skip_special_tokens=True)[0]
            
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            
            m = JSON_PAT.search(text)
            if m:
                try:
                    j = json.loads(m.group(0))
                    j["raw"] = text
                    j["prompt_id"] = i
                    results.append(j)
                except:
                    continue
                    
        except Exception:
            continue
    
    if not results:
        return {"label": "background", "reason": "no-valid-output", "raw": "parsing failed", "prompt_id": -1}
    
    # bird 판정이 있으면 우선 선택
    bird_results = [r for r in results if r.get("label") == "bird"]
    return bird_results[0] if bird_results else results[0]

def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """YOLO 라벨 파일 로드"""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append((class_id, center_x, center_y, width, height))
    return boxes

def yolo_to_bbox(center_x: float, center_y: float, width: float, height: float, 
                 img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """YOLO 정규화 좌표를 픽셀 좌표로 변환"""
    x1 = int((center_x - width / 2) * img_width)
    y1 = int((center_y - height / 2) * img_height)
    x2 = int((center_x + width / 2) * img_width)
    y2 = int((center_y + height / 2) * img_height)
    
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    return x1, y1, x2, y2

def crop_with_padding(image: Image.Image, bbox: Tuple[int, int, int, int], 
                     padding_ratio: float = 0.6) -> Image.Image:
    """바운딩 박스를 패딩과 함께 크롭"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # 작은 영역은 더 큰 패딩
    if width < 50 or height < 50:
        padding_ratio = max(padding_ratio, 1.0)
    
    pad_w = int(width * padding_ratio)
    pad_h = int(height * padding_ratio)
    
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(image.width, x2 + pad_w)
    y2_pad = min(image.height, y2 + pad_h)
    
    return image.crop((x1_pad, y1_pad, x2_pad, y2_pad))

def analyze_image(img_path: Path, label_path: Path, model, processor, 
                 resize=336, padding_ratio=0.6):
    """이미지 분석 수행"""
    try:
        # 이미지 로드
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size
        
        # YOLO 라벨 로드
        boxes = load_yolo_labels(label_path)
        
        # 1. 전체 이미지 분석
        full_img_resized = img.resize((resize, resize)) if resize > 0 else img
        full_result = run_vlm_inference(full_img_resized, model, processor, True)
        
        # 2. 크롭 영역들 분석
        crop_results = []
        for i, (class_id, center_x, center_y, width, height) in enumerate(boxes):
            bbox = yolo_to_bbox(center_x, center_y, width, height, img_width, img_height)
            cropped = crop_with_padding(img, bbox, padding_ratio)
            
            if resize > 0:
                cropped = cropped.resize((resize, resize))
            
            crop_result = run_vlm_inference(cropped, model, processor, True)
            crop_results.append({
                "box_id": i,
                "class_id": class_id,
                "bbox": list(bbox),
                "label": crop_result.get("label", "background"),
                "reason": crop_result.get("reason", ""),
                "crop_size": list(cropped.size)
            })
        
        # 3. 하이브리드 판정
        bird_crops = [c for c in crop_results if c["label"] == "bird"]
        full_is_bird = full_result.get("label") == "bird"
        
        if len(bird_crops) > 0 and full_is_bird:
            hybrid_decision = "STRONG_BIRD"
            hybrid_reason = f"Both full image and {len(bird_crops)} crops detect birds"
        elif len(bird_crops) > 0:
            hybrid_decision = "CROP_BIRD"
            hybrid_reason = f"Only crops detect birds ({len(bird_crops)} found)"
        elif full_is_bird:
            hybrid_decision = "FULL_BIRD"
            hybrid_reason = "Only full image detects bird"
        else:
            hybrid_decision = "NO_BIRD"
            hybrid_reason = "No birds detected"
        
        # 결과 반환
        return {
            "path": str(img_path),
            "full_analysis": full_result,
            "crop_analyses": crop_results,
            "hybrid_decision": hybrid_decision,
            "hybrid_reason": hybrid_reason,
            "num_detections": len(boxes),
            "num_bird_crops": len(bird_crops)
        }
        
    except Exception as e:
        print(f"분석 실패 ({img_path.name}): {e}")
        return None

def find_image_label_pairs(images_dir: Path, labels_dir: Path):
    """이미지와 라벨 파일 쌍 찾기"""
    pairs = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in img_exts:
            continue
            
        rel_path = img_path.relative_to(images_dir)
        label_name = rel_path.stem + ".txt"
        label_path = labels_dir / rel_path.parent / label_name
        
        if label_path.exists():
            pairs.append((img_path, label_path))
    
    return pairs

def organize_files_by_decision(input_dir: Path, results: List[Dict], organize_files: bool = True):
    """
    분석 결과에 따라 파일을 판정별 폴더로 정리
    
    Args:
        input_dir: 입력 폴더 경로
        results: 분석 결과 리스트
        organize_files: 파일 정리 수행 여부
    """
    if not organize_files:
        print("📌 파일 정리 건너뜀 (--no-organize 옵션)")
        return
    
    print("\n📂 파일 정리 시작...")
    
    # 폴더 확인
    original_frames_dir = input_dir / "original_frames"
    detected_frames_dir = input_dir / "detected_frames"
    labels_dir = input_dir / "labels"
    
    has_detected_frames = detected_frames_dir.exists()
    
    # 판정별 카운트
    decision_counts = {}
    for result in results:
        decision = result['hybrid_decision']
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
    
    # 판정별 폴더 생성
    for decision in decision_counts.keys():
        # original_frames
        (original_frames_dir / decision).mkdir(exist_ok=True)
        
        # detected_frames (존재하는 경우)
        if has_detected_frames:
            (detected_frames_dir / decision).mkdir(exist_ok=True)
        
        # labels
        (labels_dir / decision).mkdir(exist_ok=True)
    
    # 파일 이동
    moved_count = 0
    detected_moved_count = 0
    label_moved_count = 0
    
    for result in results:
        img_path = Path(result['path'])
        decision = result['hybrid_decision']
        
        if not img_path.exists():
            continue
        
        try:
            # 1. original_frames 이동
            target_path = original_frames_dir / decision / img_path.name
            shutil.move(str(img_path), str(target_path))
            moved_count += 1
            
            # 2. detected_frames 이동 (존재하는 경우)
            if has_detected_frames:
                detected_path = detected_frames_dir / img_path.name
                if detected_path.exists():
                    detected_target = detected_frames_dir / decision / img_path.name
                    shutil.move(str(detected_path), str(detected_target))
                    detected_moved_count += 1
            
            # 3. labels 이동
            label_filename = img_path.stem + ".txt"
            label_path = labels_dir / label_filename
            if label_path.exists():
                label_target = labels_dir / decision / label_filename
                shutil.move(str(label_path), str(label_target))
                label_moved_count += 1
                
        except Exception as e:
            print(f"  ⚠️ 파일 이동 실패 ({img_path.name}): {e}")
    
    # 정리 결과 출력
    print(f"\n✅ 파일 정리 완료!")
    print(f"📊 original_frames 이동: {moved_count}개")
    if has_detected_frames:
        print(f"📊 detected_frames 이동: {detected_moved_count}개")
    print(f"📊 labels 이동: {label_moved_count}개")
    
    print(f"\n📁 판정별 파일 분포:")
    for decision, count in decision_counts.items():
        print(f"  {decision}: {count}개")

def main(input_dir, model="llava-hf/llava-1.5-7b-hf", precision="fp16", 
         resize=336, padding_ratio=0.6, limit=0, organize_files=True):
    """하이브리드 VLM 분석 및 파일 정리 메인 함수"""
    if not DEPENDENCIES_AVAILABLE:
        print("❌ 의존성 라이브러리가 설치되지 않았습니다.")
        return
    
    # 입력 폴더 확인
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"❌ 입력 폴더가 존재하지 않습니다: {input_dir}")
        return
    
    print(f"✅ 입력 폴더: {input_dir}")
    
    # original_frames와 labels 폴더 찾기
    images_dir = input_dir / "original_frames"
    labels_dir = input_dir / "labels"
    
    # 폴더 존재 확인
    if not images_dir.exists():
        print(f"❌ original_frames 폴더가 없습니다: {images_dir}")
        return
    
    if not labels_dir.exists():
        print(f"❌ labels 폴더가 없습니다: {labels_dir}")
        return
    
    print(f"   - 이미지: {images_dir}")
    print(f"   - 라벨: {labels_dir}")
    
    # detected_frames 폴더 확인
    detected_frames_dir = input_dir / "detected_frames"
    if detected_frames_dir.exists():
        print(f"   - 검출 이미지: {detected_frames_dir}")
    
    # JSON 파일 경로 - 입력 폴더에 저장
    results_filename = "hybrid_results.json"
    results_file = input_dir / results_filename
    
    # 기존 결과 확인 및 진행 상황 분석
    existing_results = []
    processed_paths = set()
    
    if results_file.exists():
        print(f"\n📄 기존 결과 파일 발견: {results_file}")
        print("=" * 60)
        
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 처리된 파일 정보 수집
            processed_paths = {r['path'] for r in existing_results}
            
            # 진행 상황 분석
            if existing_results:
                print(f"📊 이전 작업 진행 상황:")
                print(f"   - 처리 완료: {len(existing_results)}개 이미지")
                
                # 판정별 통계
                decision_stats = Counter([r['hybrid_decision'] for r in existing_results])
                print(f"\n   판정 분포:")
                for decision, count in sorted(decision_stats.items()):
                    print(f"     • {decision}: {count}개")
                
                # 마지막 처리 파일
                last_result = existing_results[-1]
                last_path = Path(last_result['path'])
                print(f"\n   마지막 처리 파일: {last_path.name}")
                print(f"   마지막 판정: {last_result['hybrid_decision']}")
                
                # 사용자 확인
                print("\n🔄 이어서 작업을 진행합니다...")
                print("=" * 60)
                
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 파일 손상 감지: {e}")
            backup_file = results_file.with_suffix('.backup.json')
            shutil.copy2(results_file, backup_file)
            print(f"   백업 생성: {backup_file}")
            
            user_input = input("새로 시작하시겠습니까? (y/n): ")
            if user_input.lower() != 'y':
                print("작업 취소")
                return
            existing_results = []
            processed_paths = set()
    else:
        print(f"\n📄 새 작업 시작")
        print(f"   결과 파일 생성: {results_file}")
        # 빈 JSON 파일 생성
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print("=" * 60)
    
    # 모델 로드
    print("\n🤖 모델 로드 중...")
    model_obj, processor = load_model(model, precision)
    if model_obj is None:
        print("❌ 모델 로드 실패")
        return
    
    # 이미지-라벨 쌍 찾기
    all_pairs = find_image_label_pairs(images_dir, labels_dir)
    
    if not all_pairs:
        print(f"⚠️ 매칭되는 이미지-라벨 쌍을 찾을 수 없습니다.")
        return
    
    # 처리할 파일 필터링
    pairs_to_process = []
    skipped_count = 0
    
    print(f"\n📂 파일 스캔 중...")
    for img_path, label_path in all_pairs:
        if str(img_path) not in processed_paths:
            pairs_to_process.append((img_path, label_path))
        else:
            skipped_count += 1
    
    # 상태 요약
    print(f"\n📊 작업 상태:")
    print(f"   전체 파일: {len(all_pairs)}개")
    print(f"   ✅ 이미 처리됨: {skipped_count}개")
    print(f"   ⏳ 처리 대기: {len(pairs_to_process)}개")
    
    if not pairs_to_process:
        print(f"\n✨ 모든 파일이 이미 처리되었습니다!")
        
        # 최종 통계 출력
        if existing_results:
            print(f"\n📊 최종 결과:")
            decisions = [r['hybrid_decision'] for r in existing_results]
            decision_count = Counter(decisions)
            
            for decision, count in sorted(decision_count.items()):
                percentage = count/len(existing_results)*100
                print(f"   {decision}: {count}개 ({percentage:.1f}%)")
            
            # ✨ 파일 정리 옵션 추가
            if organize_files:
                print(f"\n📂 기존 결과로 파일 정리를 수행하시겠습니까? (y/n): ", end="")
                if input().lower() == 'y':
                    organize_files_by_decision(input_dir, existing_results, True)
        return
    
    # limit 적용
    if limit > 0 and limit < len(pairs_to_process):
        pairs_to_process = pairs_to_process[:limit]
        print(f"   ⚠️ limit 적용: {limit}개만 처리")
    
    # 진행률 표시 준비
    total_to_process = len(pairs_to_process)
    already_processed = len(existing_results)
    grand_total = len(all_pairs)
    
    print(f"\n🚀 분석 시작!")
    print("=" * 60)
    
    # 분석 수행
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    for i, (img_path, label_path) in enumerate(pairs_to_process):
        # 진행률 계산
        current_progress = already_processed + i + 1
        percentage = (current_progress / grand_total) * 100
        
        print(f"\n[전체 {current_progress}/{grand_total} ({percentage:.1f}%)] "
              f"[현재 세션 {i+1}/{total_to_process}]")
        print(f"처리 중: {img_path.name}")
        
        try:
            # 이미지 분석 실행
            result = analyze_image(
                img_path, label_path, model_obj, processor,
                resize, padding_ratio
            )
            
            if result:
                # 기존 결과 읽기
                with open(results_file, 'r', encoding='utf-8') as f:
                    current_results = json.load(f)
                
                # 새 결과 추가
                current_results.append(result)
                
                # 안전하게 저장 (임시 파일 사용)
                temp_file = results_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(current_results, f, ensure_ascii=False, indent=2)
                temp_file.replace(results_file)
                
                success_count += 1
                print(f"  ✅ 판정: {result['hybrid_decision']}")
                print(f"  📊 크롭 분석: {result['num_bird_crops']}/{result['num_detections']} 새 검출")
                
                # 예상 남은 시간 계산 (10개 처리 후부터)
                if i >= 9:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = (total_to_process - i - 1) * avg_time
                    
                    hours = int(remaining // 3600)
                    minutes = int((remaining % 3600) // 60)
                    seconds = int(remaining % 60)
                    
                    if hours > 0:
                        print(f"  ⏱️ 예상 남은 시간: {hours}시간 {minutes}분")
                    elif minutes > 0:
                        print(f"  ⏱️ 예상 남은 시간: {minutes}분 {seconds}초")
                    else:
                        print(f"  ⏱️ 예상 남은 시간: {seconds}초")
                        
            else:
                failed_count += 1
                print(f"  ❌ 분석 실패")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자에 의해 중단됨")
            print(f"💾 {success_count}개 결과가 저장되었습니다.")
            print("다시 실행하면 이어서 작업할 수 있습니다.")
            return
            
        except Exception as e:
            failed_count += 1
            print(f"  ❌ 오류 발생: {e}")
            continue
    
    # 작업 완료
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✨ 현재 세션 작업 완료!")
    print(f"   처리 시간: {elapsed_total/60:.1f}분")
    print(f"   성공: {success_count}개")
    print(f"   실패: {failed_count}개")
    
    # 최종 결과 로드
    with open(results_file, 'r', encoding='utf-8') as f:
        final_results = json.load(f)
    
    print(f"\n📊 전체 누적 결과:")
    print(f"   총 처리된 파일: {len(final_results)}/{len(all_pairs)}개")
    
    # 최종 판정 분포
    if final_results:
        decisions = [r['hybrid_decision'] for r in final_results]
        decision_count = Counter(decisions)
        
        print(f"\n🎯 전체 판정 분포:")
        for decision, count in sorted(decision_count.items()):
            percentage = count/len(final_results)*100
            print(f"   {decision}: {count}개 ({percentage:.1f}%)")
    
    # 파일 정리 수행
    if organize_files and success_count > 0:
        print(f"\n📂 파일 정리를 시작하시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            # 이번에 처리된 결과만 정리
            new_results = final_results[-success_count:]
            organize_files_by_decision(input_dir, new_results, True)
    
    print("\n✅ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='하이브리드 VLM 분석 및 파일 정리')
    parser.add_argument('input_dir', type=str, help='입력 폴더 경로')
    parser.add_argument('--model', type=str, default="llava-hf/llava-1.5-7b-hf", 
                       help='VLM 모델 (기본: llava-hf/llava-1.5-7b-hf)')
    parser.add_argument('--precision', type=str, default="fp16", 
                       choices=["bf16", "fp16", "fp32"],
                       help='모델 정밀도 (기본: fp16)')
    parser.add_argument('--resize', type=int, default=336, 
                       help='VLM 입력 크기 (기본: 336)')
    parser.add_argument('--padding', type=float, default=0.6, 
                       help='크롭 패딩 비율 (기본: 0.6)')
    parser.add_argument('--limit', type=int, default=0, 
                       help='처리할 최대 이미지 수, 0=전체 (기본: 0)')
    parser.add_argument('--no-organize', action='store_true', 
                       help='파일 정리 건너뛰기 (JSON만 생성)')
    
    args = parser.parse_args()
    
    # 실행
    main(
        input_dir=args.input_dir,
        model=args.model,
        precision=args.precision,
        resize=args.resize,
        padding_ratio=args.padding,
        limit=args.limit,
        organize_files=not args.no_organize
    )
