import os
import argparse
import glob
from pathlib import Path
import cv2
from datetime import datetime

# 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDNN_BENCHMARK'] = 'True'

from ultralytics import YOLO
import torch
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from tqdm import tqdm
import numpy as np

def calculate_box_center(box):
    """바운딩 박스의 중심점 계산"""
    x1, y1, x2, y2 = box[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y

def calculate_distance(center1, center2):
    """두 중심점 사이의 유클리드 거리 계산"""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def is_static_object(current_boxes, previous_detections, frame_idx, threshold_distance=30, min_consecutive_frames=3):
    """
    현재 프레임의 박스들이 정적 객체인지 판단
    
    Args:
        current_boxes: 현재 프레임의 박스들 (텐서 또는 numpy 배열)
        previous_detections: 이전 프레임들의 탐지 기록 (frame_idx를 키로 하는 딕셔너리)
        frame_idx: 현재 프레임 인덱스
        threshold_distance: 같은 위치로 판단할 거리 임계값 (픽셀)
        min_consecutive_frames: 필터링할 최소 연속 프레임 수
    
    Returns:
        filtered_boxes: 필터링된 박스들의 인덱스 리스트
    """
    # current_boxes가 텐서인 경우 numpy로 변환
    if torch.is_tensor(current_boxes):
        current_boxes = current_boxes.cpu().numpy()
    
    if frame_idx < min_consecutive_frames - 1:
        # 충분한 이전 프레임이 없으면 모든 박스 유지
        return list(range(len(current_boxes)))
    
    filtered_boxes = []
    
    for box_idx, current_box in enumerate(current_boxes):
        # current_box도 numpy 배열로 확실히 변환
        if torch.is_tensor(current_box):
            current_box = current_box.cpu().numpy()
            
        current_center = calculate_box_center(current_box)
        is_static = True
        
        # 이전 연속 프레임들에서 비슷한 위치의 박스가 있는지 확인
        consecutive_count = 0
        
        for prev_frame_offset in range(1, min_consecutive_frames):
            prev_frame_idx = frame_idx - prev_frame_offset
            
            if prev_frame_idx not in previous_detections:
                is_static = False
                break
            
            # 이전 프레임에서 비슷한 위치의 박스 찾기
            found_similar = False
            for prev_box in previous_detections[prev_frame_idx]:
                prev_center = calculate_box_center(prev_box)
                distance = calculate_distance(current_center, prev_center)
                
                if distance < threshold_distance:
                    found_similar = True
                    consecutive_count += 1
                    break
            
            if not found_similar:
                is_static = False
                break
        
        # 연속된 프레임에서 모두 비슷한 위치에 있으면 정적 객체로 판단
        if not is_static or consecutive_count < min_consecutive_frames - 1:
            filtered_boxes.append(box_idx)
    
    return filtered_boxes

def parse_args():
    # 첫 번째 인자를 확인하여 모드 결정
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'combine':
        # combine 모드
        parser = argparse.ArgumentParser(description='탐지 결과 CSV 파일들 통합')
        parser.add_argument('mode', choices=['combine'], help='실행 모드')
        parser.add_argument('--run_dir', '-r', type=str, required=True,
                            help='결과가 저장된 runs/detect 하위 폴더 경로')
        return parser.parse_args()
    else:
        # 기본 detect 모드
        parser = argparse.ArgumentParser(description='YOLO를 사용한 동영상 객체 탐지')
        parser.add_argument('--input_dir', '-i', type=str, required=True,
                            help='입력 동영상 파일 또는 폴더 경로')
        parser.add_argument('--run_name', '-n', type=str, default=None,
                            help='실행 이름 (기본값: detect_YYYYMMDD_HHMMSS 형식으로 자동 생성)')
        parser.add_argument('--model', '-m', type=str, default='./pt_model/FLY_37LB4.pt',
                            help='YOLO 모델 파일 경로 (기본값: ./pt_model/FLY_37LB4.pt)')
        parser.add_argument('--conf', type=float, default=0.25,
                            help='신뢰도 임계값 (기본값: 0.25)')
        parser.add_argument('--iou', type=float, default=0.7,
                            help='NMS IoU 임계값 (기본값: 0.7)')
        parser.add_argument('--device', type=str, default='0',
                            help='사용할 디바이스 (0: GPU, cpu: CPU)')
        parser.add_argument('--vid_stride', type=int, default=10,
                            help='비디오 프레임 스트라이드 (기본값: 10)')
        parser.add_argument('--start', '-s', type=int, default=None,
                            help='처리할 동영상의 시작 인덱스 (0부터 시작)')
        parser.add_argument('--end', '-e', type=int, default=None,
                            help='처리할 동영상의 끝 인덱스 (포함)')
        parser.add_argument('--worker', '-w', type=int, default=2,
                            help='동시에 처리할 스레드 수 (기본값: 2)')
        parser.add_argument('--reverse', '-r', action='store_true',
                            help='동영상 파일을 내림차순으로 정렬하여 처리')
        parser.add_argument('--filter_static', action='store_true',
                            help='연속된 프레임에서 같은 위치의 정적 객체 필터링 (기본값: False)')
        parser.add_argument('--static_threshold', type=float, default=30.0,
                            help='정적 객체 판단을 위한 거리 임계값 (픽셀, 기본값: 30.0)')
        parser.add_argument('--static_frames', type=int, default=3,
                            help='정적 객체로 판단할 최소 연속 프레임 수 (기본값: 3)')
        return parser.parse_args()

def get_video_files(input_path):
    """입력 경로에서 동영상 파일들을 찾아 반환 (하위 폴더 포함)"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        # 단일 파일인 경우
        if input_path.suffix.lower() in video_extensions:
            video_files.append(str(input_path))
    elif input_path.is_dir():
        # 폴더인 경우 재귀적으로 모든 동영상 파일 찾기
        for ext in video_extensions:
            # ** 패턴을 사용하여 모든 하위 디렉토리 검색
            pattern = f'**/*{ext}'
            found_files = list(input_path.glob(pattern))
            # 대소문자 구분 없이 검색
            found_files.extend(list(input_path.glob(pattern.upper())))
            
            # 중복 제거하고 문자열로 변환
            for file in found_files:
                file_str = str(file)
                if file_str not in video_files:
                    video_files.append(file_str)
    
    # 파일 경로로 정렬하여 일관된 순서 보장
    video_files.sort()
    
    return video_files

# 전역 변수로 lock과 처리 정보 관리
print_lock = Lock()
process_counter = {'completed': 0, 'total': 0}
position_manager = {'current': 0, 'lock': Lock()}

def process_video(video_path, model_path, run_dir, args, input_base_dir, video_index):
    """YOLO 형식에 맞춰 파일을 저장하는 개선된 버전"""
    video_path = Path(video_path)
    video_name = video_path.stem
    
    # 동영상의 총 프레임 수 가져오기
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # 각 스레드에서 모델 로드
    model = YOLO(model_path)
    
    # 디렉토리 생성
    frames_dir = Path(run_dir) / 'detected_frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    original_frames_dir = Path(run_dir) / 'original_frames'
    original_frames_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dir = Path(run_dir) / 'csv_results'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO 실행
    results = model.predict(
        source=video_path,
        save=False,
        save_txt=True,
        save_conf=True,
        project='runs/detect',
        name=Path(run_dir).name,
        exist_ok=True,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        device=args.device,
        stream=True,
        verbose=False,
        vid_stride=args.vid_stride,
        line_width=2,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        retina_masks=False,
        classes=None,
    )
    
    # 결과 처리
    all_detections = []
    saved_frames = set()
    previous_detections = {}
    filtered_count = 0
    
    # 진행률 표시
    effective_frames = total_frames // args.vid_stride
    display_name = video_name[:30] + "..." if len(video_name) > 30 else video_name
    
    with position_manager['lock']:
        my_position = position_manager['current']
        position_manager['current'] += 1
    
    progress_bar = tqdm(
        total=effective_frames,
        desc=f"[{video_index:02d}] {display_name}",
        unit="프레임",
        position=my_position,
        leave=True,
        ncols=120,
        bar_format='{desc:>40}: {percentage:3.0f}%|{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour='green'
    )
    
    # 여기가 핵심 부분입니다!
    # frame_idx는 YOLO가 사용하는 인덱스와 동일합니다
    for frame_idx, result in enumerate(results):
        progress_bar.update(1)
        boxes = result.boxes
        
        # 실제 프레임 번호 계산
        actual_frame = frame_idx * args.vid_stride
        
        if boxes is not None and boxes.data.shape[0] > 0:
            # 정적 객체 필터링 로직 (기존과 동일)
            if args.filter_static:
                valid_box_indices = is_static_object(
                    boxes.data, 
                    previous_detections, 
                    frame_idx,
                    threshold_distance=args.static_threshold,
                    min_consecutive_frames=args.static_frames
                )
                filtered_count += len(boxes.data) - len(valid_box_indices)
                
                if len(valid_box_indices) > 0:
                    valid_boxes = [boxes.data[i] for i in valid_box_indices]
                else:
                    valid_boxes = []
            else:
                valid_boxes = boxes.data
                valid_box_indices = list(range(boxes.data.shape[0]))
            
            # 현재 프레임의 박스 정보 저장
            current_frame_boxes = []
            for box in boxes.data:
                current_frame_boxes.append(box.cpu().numpy())
            previous_detections[frame_idx] = current_frame_boxes
            
            # 오래된 프레임 정보 삭제
            if len(previous_detections) > 10:
                oldest_frame = min(previous_detections.keys())
                del previous_detections[oldest_frame]
            
            # 필터링된 박스가 있는 경우에만 저장
            if len(valid_boxes) > 0:
                if actual_frame not in saved_frames:
                    # 여기가 수정된 부분입니다!
                    # YOLO와 동일한 형식으로 저장: video_name_frame_idx
                    original_frame = result.orig_img
                    
                    # YOLO 형식: {video_name}_{frame_idx}.jpg
                    original_frame_filename = original_frames_dir / f'{video_name}_{frame_idx}.jpg'
                    cv2.imwrite(str(original_frame_filename), original_frame)
                    
                    # 바운딩 박스가 그려진 프레임도 동일한 형식으로
                    annotated_frame = result.plot()
                    frame_filename = frames_dir / f'{video_name}_{frame_idx}.jpg'
                    cv2.imwrite(str(frame_filename), annotated_frame)
                    
                    saved_frames.add(actual_frame)
                
                # CSV 데이터 수집 (기존과 동일)
                for box_idx in valid_box_indices:
                    box = boxes.data[box_idx]
                    x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                    
                    detection = {
                        'video_name': video_name,
                        'frame': actual_frame,  # CSV에는 실제 프레임 번호 저장
                        'frame_idx': frame_idx,  # YOLO 인덱스도 함께 저장
                        'class_name': model.names[int(cls)],
                        'confidence': float(conf),
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                        'center_x': float((x1 + x2) / 2),
                        'center_y': float((y1 + y2) / 2),
                    }
                    all_detections.append(detection)
    
    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(all_detections)
    
    # 결과 출력
    with print_lock:
        if args.filter_static:
            print(f"\n[{video_index:02d}] {video_name}: 총 {len(all_detections)}개 객체 탐지됨 (정적 객체 {filtered_count}개 필터링됨)")
        else:
            print(f"\n[{video_index:02d}] {video_name}: 총 {len(all_detections)}개 객체 탐지됨")
    
    # CSV 저장
    csv_path = csv_dir / f'{video_name}_detections.csv'
    df.to_csv(csv_path, index=False)
    
    # 파일 크기 확인
    if csv_path.exists():
        file_size = csv_path.stat().st_size
        with print_lock:
            print(f"    CSV 저장 완료: {csv_path.name} ({file_size} bytes)")
    
    # 프로그레스 바 완료
    progress_bar.colour = 'blue'
    progress_bar.set_description(f"[{video_index:02d}] {display_name} ✓")
    progress_bar.refresh()
    progress_bar.close()
    
    with print_lock:
        process_counter['completed'] += 1
    
    return df

def combine_all_results(run_dir, print_stats=True):
    """runs/detect/{run_name}/csv_results 안의 모든 CSV 파일을 읽어서 통합
    
    Args:
        run_dir: 결과가 저장된 runs/detect 하위 디렉토리
        print_stats: 통계 정보 출력 여부
        
    Returns:
        combined_df: 통합된 DataFrame (없으면 None)
        csv_files: 찾은 CSV 파일 리스트
    """
    run_dir = Path(run_dir)
    csv_dir = run_dir / 'csv_results'
    csv_files = list(csv_dir.glob('*_detections.csv'))
    
    if not csv_files:
        if print_stats:
            print(f"\n⚠️  {csv_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return None, []
    
    if print_stats:
        print(f"\n📂 {len(csv_files)}개의 CSV 파일을 통합합니다...")
    
    all_dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
        except Exception as e:
            if print_stats:
                print(f"⚠️  {csv_file} 읽기 실패: {e}")
    
    if not all_dataframes:
        if print_stats:
            print(f"⚠️  통합할 수 있는 유효한 CSV 파일이 없습니다.")
        return None, csv_files
    
    # DataFrame 통합
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 결과 저장 - runs/detect/{run_name} 디렉토리에 저장
    combined_csv = run_dir / 'all_detections_combined.csv'
    combined_df.to_csv(combined_csv, index=False)
    
    if print_stats:
        print(f"✅ 통합 완료: {combined_csv}")
        print(f"\n📊 통합 결과 요약:")
        print(f"   - 총 탐지 객체: {len(combined_df):,}개")
        print(f"   - 통합된 동영상: {combined_df['video_name'].nunique()}개")
        
        # 클래스별 탐지 수
        if 'class_name' in combined_df.columns and len(combined_df) > 0:
            print(f"\n📋 전체 클래스별 탐지 수:")
            class_counts = combined_df['class_name'].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(combined_df)) * 100
                print(f"   - {class_name}: {count:,}개 ({percentage:.1f}%)")
            
            # 상위 10개 클래스만 보여주기 (클래스가 많은 경우)
            if len(class_counts) > 10:
                print(f"\n   (총 {len(class_counts)}개 클래스 중 상위 10개만 표시)")
    
    return combined_df, csv_files


def main():
    args = parse_args()
    
    # PyTorch 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    torch.cuda.empty_cache()
    
    # run_name 생성 (지정하지 않은 경우 타임스탬프 사용)
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"detect_{timestamp}"
    else:
        run_name = args.run_name
    
    # runs/detect 아래에 실행별 디렉토리 생성
    run_dir = Path('runs/detect') / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 결과 저장 위치: {run_dir}")
    
    # 실행 정보를 저장 (나중에 참고할 수 있도록)
    run_info = {
        'run_name': run_name,
        'timestamp': datetime.now().isoformat(),
        'input_dir': args.input_dir,
        'model': args.model,
        'conf': args.conf,
        'iou': args.iou,
        'vid_stride': args.vid_stride,
        'filter_static': args.filter_static,
        'static_threshold': args.static_threshold if args.filter_static else None,
        'static_frames': args.static_frames if args.filter_static else None,
        'device': args.device,
        'worker': args.worker
    }
    
    # 실행 정보를 JSON으로 저장
    with open(run_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    
    # 입력 동영상 파일들 찾기
    video_files = get_video_files(args.input_dir)
    
    if not video_files:
        print(f"❌ {args.input_dir}에서 동영상 파일을 찾을 수 없습니다.")
        return
    
    print(f"📁 총 {len(video_files)}개의 동영상 파일을 찾았습니다.")
    
    # reverse 옵션이 설정된 경우 파일 리스트를 뒤집기
    if args.reverse:
        video_files.reverse()
        print(f"📌 동영상 파일을 내림차순으로 정렬했습니다.")
    
    # start와 end 인자에 따라 파일 리스트 슬라이싱
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end + 1 if args.end is not None else len(video_files)
        
        # 인덱스 범위 검증
        if start_idx < 0:
            start_idx = 0
        if end_idx > len(video_files):
            end_idx = len(video_files)
        
        if start_idx >= end_idx:
            print(f"❌ 잘못된 범위: start({args.start})가 end({args.end})보다 크거나 같습니다.")
            return
        
        # 원본 파일 리스트와 슬라이싱된 리스트 정보 출력
        print(f"📌 선택된 범위: 인덱스 {start_idx}부터 {end_idx-1}까지")
        video_files = video_files[start_idx:end_idx]
        print(f"✅ 처리할 동영상: {len(video_files)}개")
        
        # 선택된 파일 목록 출력
        sort_order = "(내림차순)" if args.reverse else "(오름차순)"
        print(f"\n선택된 동영상 파일 {sort_order}:")
        base_path = Path(args.input_dir)
        for idx, file in enumerate(video_files, start=start_idx):
            # 상대 경로 표시
            try:
                relative_path = Path(file).relative_to(base_path)
                print(f"  [{idx}] {relative_path}")
            except ValueError:
                # 상대 경로를 만들 수 없는 경우 파일명만 표시
                print(f"  [{idx}] {Path(file).name}")
    
    # 전역 카운터 설정
    process_counter['total'] = len(video_files)
    process_counter['completed'] = 0
    position_manager['current'] = 0
    
    print(f"\n🚀 {args.worker}개의 스레드로 처리를 시작합니다.")
    print(f"모델: {args.model}")
    
    # 정적 객체 필터링 설정 출력
    if args.filter_static:
        print(f"🔍 정적 객체 필터링 활성화:")
        print(f"   - 거리 임계값: {args.static_threshold} 픽셀")
        print(f"   - 최소 연속 프레임: {args.static_frames}개")
        print(f"   ⚠️  연속 {args.static_frames}프레임 이상 같은 위치({args.static_threshold}px 이내)에 나타나는 객체는 제거됩니다.")
    else:
        print(f"🔍 정적 객체 필터링: 비활성화 (--filter_static 옵션으로 활성화 가능)")
    
    # GPU 사용 시 멀티스레드 경고
    if args.worker > 1 and args.device != 'cpu':
        print(f"⚠️  주의: GPU 사용 시 {args.worker}개의 스레드가 GPU 메모리를 공유합니다.")
        print(f"   메모리 부족 시 --worker 수를 줄이거나 --device cpu를 사용하세요.")
    
    print(f"\n📊 각 동영상의 처리 진행률이 아래에 표시됩니다:")
    print("="*100)
    
    start_time = time.time()
    
    # ThreadPoolExecutor를 사용한 병렬 처리
    try:
        with ThreadPoolExecutor(max_workers=args.worker) as executor:
            # 각 비디오 파일에 대한 future 생성
            future_to_video = {
                executor.submit(process_video, video_file, args.model, run_dir, args, args.input_dir, idx+1): (video_file, idx)
                for idx, video_file in enumerate(video_files)
            }
            
            # 완료된 작업들 처리
            for future in as_completed(future_to_video):
                video_file, idx = future_to_video[future]
                try:
                    df = future.result()
                    # 개별 비디오 처리 완료 (CSV는 이미 저장됨)
                except Exception as e:
                    with print_lock:
                        print(f"❌ {video_file} 처리 중 오류 발생: {e}")
                        import traceback
                        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단됨. 진행중인 작업을 종료합니다...")
        executor.shutdown(wait=False)
        return
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 모든 프로그레스 바가 표시된 후에 통계 출력
    print("\n" * (position_manager['current'] + 2))
    print("="*100)
    print(f"\n🎉 모든 처리 완료!")
    print(f"📊 전체 통계:")
    print(f"   - 처리된 동영상: {len(video_files)}개")
    print(f"   - 총 처리 시간: {processing_time:.2f}초 (평균: {processing_time/len(video_files):.2f}초/영상)")
    print(f"   - 사용된 스레드 수: {args.worker}개")
    print(f"   - 결과 저장 위치: {run_dir}")
    
    # 자동으로 CSV 파일들 통합
    print("\n📊 CSV 파일 통합을 시작합니다...")
    combined_df, csv_files = combine_all_results(run_dir)
    
    print(f"\n✅ 모든 작업이 완료되었습니다!")
    print(f"📁 결과 확인:")
    print(f"   - 탐지된 프레임 이미지: {run_dir}/detected_frames/")
    print(f"   - 개별 CSV 파일들: {run_dir}/csv_results/")
    print(f"   - 통합 CSV 파일: {run_dir}/all_detections_combined.csv")
    print(f"   - YOLO 텍스트 파일: {run_dir}/labels/")

if __name__ == '__main__': 
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'combine':
        # 결과 통합만 실행
        args = parse_args()
        # runs/detect 아래의 특정 폴더에서 결과 통합
        run_dir = Path('runs/detect') / args.run_dir
        if not run_dir.exists():
            print(f"❌ {run_dir} 디렉토리를 찾을 수 없습니다.")
        else:
            combined_df, csv_files = combine_all_results(run_dir)
    else:
        # 기본 동작: 객체 탐지 실행
        main()
