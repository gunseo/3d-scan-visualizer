import trimesh
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
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

def cartesian_to_spherical_numpy(points_xyz, origin):
    """
    (X, Y, Z) 좌표 배열을 지정된 원점 기준의 (distance, azimuth, elevation)으로 변환합니다.
    """
    # 1. 모든 점을 스캐너 원점 기준으로 이동
    translated_points = points_xyz - origin
    
    # 2. 거리(distance) 계산
    distance = np.linalg.norm(translated_points, axis=1)
    
    # x, y, z 성분 분리
    x = translated_points[:, 0]
    y = translated_points[:, 1]
    z = translated_points[:, 2]
    
    # 3. 방위각(azimuth) 계산 (XY 평면에서의 각도)
    azimuth_rad = np.arctan2(y, x)
    
    # 4. 고도각(elevation) 계산 (XY 평면과 Z축 사이의 각도)
    d_xy = np.sqrt(x**2 + y**2)
    elevation_rad = np.arctan2(z, d_xy)
    
    # 5. 라디안을 도로 변환
    azimuth_deg = np.degrees(azimuth_rad)
    elevation_deg = np.degrees(elevation_rad)
    
    # (distance, azimuth, elevation) 형태로 열을 합침
    return np.stack([distance, azimuth_deg, elevation_deg], axis=1)


def sample_and_convert_to_spherical(stl_file_path, output_file_path, num_points):
    """STL 표면에 균일하게 점을 찍고, 그 결과를 distance, azimuth, elevation으로 변환하여 저장합니다."""
    # 1. STL 파일 불러오기
    try:
        mesh = trimesh.load_mesh(stl_file_path)
        print(f"'{os.path.basename(stl_file_path)}' 파일을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 2. [핵심] 면적 기반 무작위 샘플링으로 표면에 균일하게 점 생성 (결과는 X, Y, Z 좌표)
    print(f"{num_points}개의 점을 균일하게 샘플링합니다...")
    sampled_points_xyz, _ = mesh.sample(count=num_points, return_index=True)

    if sampled_points_xyz.size == 0:
        print("경고: 샘플링된 점이 하나도 없습니다.")
        return

    # 3. 스캐너 원점 위치 설정 (기존 코드와 동일)
    bounds = mesh.bounds
    model_height = bounds[1][2] - bounds[0][2]
    offset_height = model_height * 0.05 
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_y = (bounds[0][1] + bounds[1][1]) / 2.0
    bottom_z = bounds[0][2]
    scanner_origin = np.array([center_x, center_y, bottom_z + offset_height])
    print(f"스캐너 기준점: {np.round(scanner_origin, 2)}")

    # 4. [핵심] 샘플링된 (X, Y, Z) 점들을 (distance, azimuth, elevation)으로 역변환
    print("샘플링된 점들을 구면 좌표계로 변환합니다...")
    scan_data = cartesian_to_spherical_numpy(sampled_points_xyz, scanner_origin)
    
    print(f"변환 완료! 총 {len(scan_data)}개의 데이터를 생성했습니다.")
    
    # 5. CSV 파일로 저장
    header = ['distance', 'azimuth', 'elevation']
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(scan_data)
        print(f"변환된 데이터를 '{os.path.basename(output_file_path)}' 파일로 저장했습니다.")
    except Exception as e:
        print(f"파일 저장 중 오류가 발생했습니다: {e}")


# --- 코드 실행 ---
if __name__ == '__main__':
    stl_file = select_file(
        title="STL 파일을 선택하세요",
        filetypes=[("STL files", "*.stl")]
    )
    
    if stl_file:
        root = tk.Tk()
        root.withdraw()
        num_points_str = simpledialog.askstring("점 개수 입력", "생성할 점의 개수를 입력하세요:", parent=root)
        
        if num_points_str:
            try:
                num_points = int(num_points_str)
                output_csv_file = select_save_path()
                
                if output_csv_file:
                    sample_and_convert_to_spherical(stl_file, output_csv_file, num_points)
                else:
                    print("저장 경로가 선택되지 않았습니다. 프로그램을 종료합니다.")
            except (ValueError, TypeError):
                print("잘못된 숫자 형식입니다. 프로그램을 종료합니다.")
        else:
            print("점 개수가 입력되지 않았습니다. 프로그램을 종료합니다.")
    else:
        print("STL 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")