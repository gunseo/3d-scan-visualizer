import pandas as pd
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import os

# =============================================================================
# 1. 분석 파라미터
# =============================================================================
# Azimuth별 평균 거리가 이 값(mm) 이상 변하면 '급변(Jump)'으로 판단합니다.
JUMP_DETECTION_THRESHOLD = 235.0

# =============================================================================
# 2. 3D 좌표 계산 함수
# =============================================================================
def calculate_xyz(df):
    """dist, az, el 값을 x, y, z 좌표로 변환하고 편심 오프셋을 적용합니다."""
    
    # ▼▼▼ [핵심 변경] 이 함수 내에서 el < 180 조건으로 데이터를 먼저 필터링합니다. ▼▼▼
    df_proc = df[df['el'] < 180].copy()
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    scanner_offset = 58.2  # 편심 회전 오프셋 값 (mm)
    
    # az, el 순으로 정렬하여 계산 일관성 유지 (필터링된 데이터 사용)
    df_sorted = df_proc.sort_values(by=['az', 'el'])
    
    # 정렬된 데이터를 기반으로 3D 좌표 계산
    dist = df_sorted['dist'].values
    az_rad = np.radians(df_sorted['az'].values)
    el_rad = np.radians(df_sorted['el'].values + 90)

    # 1. 측정 헤드로부터의 상대 좌표
    x_relative = dist * np.cos(el_rad) * np.sin(az_rad)
    y_relative = dist * np.cos(el_rad) * np.cos(az_rad)
    z_relative = dist * np.sin(el_rad)

    # 2. Azimuth 회전에 따른 측정 헤드의 위치 (오프셋 보정)
    scanner_x = scanner_offset * np.cos(az_rad)
    scanner_y = scanner_offset * np.sin(az_rad)

    # 3. 최종 좌표 = 측정 헤드 위치 + 상대 좌표
    df_sorted['x'] = scanner_x + x_relative
    df_sorted['y'] = scanner_y - y_relative
    df_sorted['z'] = z_relative
    
    return df_sorted

# =============================================================================
# 3. 평균 거리 기반 급변 지점 탐색 함수
# =============================================================================
def find_valid_azimuth_ranges(df, threshold):
    """
    지정된 Elevation 범위의 점들만 사용하여 평균 거리 급변 지점을 찾고,
    유효한 Azimuth 범위를 반환합니다.
    """
    print(f"\n[분석 시작] Elevation 0-90°, 270-360° 범위의 점들로 분석합니다.")
    
    # 1. Elevation 필터링 적용
    df_filtered = df[(df['el'] <= 180)].copy()
    
    if df_filtered.empty:
        print("-> 분석할 데이터가 없습니다.")
        return None

    # 2. Azimuth별 평균 거리 계산
    avg_dist_by_az = df_filtered.groupby('az')['dist'].mean().sort_index()

    if len(avg_dist_by_az) < 2:
        print("-> Azimuth 종류가 부족하여 분석할 수 없습니다.")
        return None
        
    # 3. Forward Scan (0도부터 증가)
    forward_diffs = avg_dist_by_az.diff().abs()
    jumps_forward = forward_diffs[forward_diffs > threshold].index.tolist()
    
    # 5. 최종 유효 범위 결정
    min_az = df['az'].min()
    end_az_forward = jumps_forward[0] if jumps_forward else df['az'].max()
    
    print(f"-> Forward 유효 Azimuth 범위: {min_az:.2f}° ~ {end_az_forward:.2f}°")
    
    return (min_az, end_az_forward)

# =============================================================================
# 4. 메인 실행부
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    
    print("분석할 CSV 파일 1개를 선택하세요.")
    filepath = filedialog.askopenfilename(title="CSV 파일 선택", filetypes=[("CSV Files", "*.csv")])

    if not filepath:
        print("파일이 선택되지 않아 프로그램을 종료합니다.")
    else:
        print(f"\n[파일 로딩] '{os.path.basename(filepath)}'")
        try:
            # 1. CSV 파일 로드
            master_df = pd.read_csv(filepath, header=None, names=['dist', 'az', 'el'])

            # 2. 유효 Azimuth 범위 분석
            range_forward = find_valid_azimuth_ranges(master_df, JUMP_DETECTION_THRESHOLD)

            # 3. 3D 좌표 계산
            pcd_df = calculate_xyz(master_df)
            
            # 4. 시각화를 위한 색상 결정
            print("\n[색상 적용] 유효 Azimuth와 Elevation 범위에 따라 점들을 색칠합니다.")
            
            # 기본 색상은 회색
            num_points = len(pcd_df)
            colors = np.full((num_points, 3), [0.5, 0.5, 0.5]) # RGB: 회색
            
            if range_forward:
                # --- ▼▼▼ 색상 마스크 로직 수정 ▼▼▼ ---
                
                # 조건 1: Azimuth가 유효 범위 내에 있는가?
                azimuth_mask = (pcd_df['az'] >= range_forward[0]) & (pcd_df['az'] <= range_forward[1])
                
                # 조건 2: Elevation이 유효 범위 내에 있는가?
                elevation_mask = (pcd_df['el'] <= 180)
                
                # 최종 마스크: 두 조건을 모두 만족하는 점들만 선택
                valid_mask = azimuth_mask & elevation_mask
                
                # --- ▲▲▲ 로직 수정 완료 ▲▲▲ ---

                # 해당 점들을 초록색으로 변경
                colors[valid_mask] = [0.0, 1.0, 0.0] # RGB: 초록색
                print(f"-> 총 {np.sum(valid_mask)}개의 점이 초록색으로 표시됩니다.")

            # 5. Open3D로 시각화
            print("\n[3D 시각화] 뷰어 창을 확인하세요.")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_df[['x', 'y', 'z']].values)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            o3d.visualization.draw_geometries([pcd], window_name="Azimuth & Elevation 유효 구간 시각화")
            
            print("\n✨ 모든 작업이 완료되었습니다.")

        except Exception as e:
            print(f"오류 발생: {e}")