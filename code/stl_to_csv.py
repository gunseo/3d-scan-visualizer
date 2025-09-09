import trimesh
import numpy as np
import tkinter as tk
from tkinter import filedialog
import csv
import os

def select_file(title, filetypes):
    """파일 탐색기를 열어 사용자가 파일을 선택하도록 하는 범용 함수."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def select_save_path():
    """'다른 이름으로 저장' 탐색기를 열어 CSV 저장 경로를 지정하게 합니다."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        title="CSV 파일 저장 위치를 선택하세요",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    return file_path

def simulate_scan(stl_file_path, output_file_path, noise_level=0.0):
    """
    STL 파일을 불러와 distance, azimuth, elevation 값을 CSV 파일로 저장합니다.
    noise_level 파라미터를 통해 거리 값에 가우시안 노이즈를 추가할 수 있습니다.
    """
    # 1. STL 파일 불러오기 (이전과 동일)
    try:
        mesh = trimesh.load_mesh(stl_file_path)
        print(f"'{os.path.basename(stl_file_path)}' 파일을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 2. 스캐너 위치 설정 (이전과 동일)
    bounds = mesh.bounds
    model_height = bounds[1][2] - bounds[0][2]
    offset_height = model_height * 0.05
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_y = (bounds[0][1] + bounds[1][1]) / 2.0
    bottom_z = bounds[0][2]
    ray_origin = np.array([center_x, center_y, bottom_z + offset_height])
    print(f"스캐너 위치를 모델 바닥면에서 {offset_height:.2f}만큼 위인 {np.round(ray_origin, 2)}(으)로 설정했습니다.")
    
    scan_data = []

    print("스캔 시뮬레이션을 시작합니다...")
    # 3. 두 개의 모터 회전 시뮬레이션 (이전과 동일)
    for azimuth in np.arange(0, 360, 1):
        for elevation in np.arange(0, 360, 1):
            
            # 4. 광선 방향 벡터 계산 (이전과 동일)
            azimuth_rad = np.deg2rad(azimuth)
            elevation_rad = np.deg2rad(elevation)
            x = np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = np.sin(elevation_rad)
            ray_direction = [x, y, z]

            # 5. 광선과 메시의 교차점 찾기 (이전과 동일)
            locations, _, _ = mesh.ray.intersects_location(
                ray_origins=[ray_origin],
                ray_directions=[ray_direction]
            )
            
            if len(locations) > 0:
                point = locations[0]
                distance = np.linalg.norm(point - ray_origin)
                
                # --- ✨ 노이즈 추가 부분 ✨ ---
                if noise_level > 0:
                    # 평균이 0이고 표준편차가 noise_level인 정규분포에서 노이즈 생성
                    noise = np.random.normal(loc=0.0, scale=noise_level)
                    distance += noise
                
                # 노이즈로 인해 거리가 0 미만이 되는 것을 방지
                if distance < 0:
                    distance = 0.0
                # -------------------------

                scan_data.append([distance, azimuth, elevation])

    if not scan_data:
        print("경고: 스캔된 점이 하나도 없습니다.")
        return

    print(f"스캔 완료! 총 {len(scan_data)}개의 데이터를 찾았습니다.")
    
    # 6. CSV 파일로 저장 (이전과 동일)
    header = ['distance', 'azimuth', 'elevation']
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(scan_data)
        print(f"스캔 데이터를 '{os.path.basename(output_file_path)}' 파일로 저장했습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")


# --- 코드 실행 ---
if __name__ == '__main__':
    stl_file = select_file(
        title="스캔할 STL 파일을 선택하세요",
        filetypes=[("STL files", "*.stl")]
    )
    
    if stl_file:
        output_csv_file = select_save_path()
        if output_csv_file:
            # --- ✨ 노이즈 강도 설정 ✨ ---
            # 이 값을 조절하여 노이즈의 크기를 변경할 수 있습니다.
            # 단위는 STL 파일의 단위와 동일합니다 (예: mm).
            NOISE_STRENGTH = 10
            
            print(f"거리 값에 표준편차 {NOISE_STRENGTH}의 노이즈를 추가합니다.")
            simulate_scan(stl_file, output_csv_file, noise_level=NOISE_STRENGTH)
            # -------------------------
        else:
            print("저장 경로가 선택되지 않았습니다. 프로그램을 종료합니다.")
    else:
        print("STL 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")