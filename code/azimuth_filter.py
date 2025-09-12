import pandas as pd
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# =============================================================================
# 0. (추가) 디버깅을 위한 샘플 데이터 생성 함수
# =============================================================================
def generate_sample_data():
    """테스트를 위한 샘플 데이터프레임을 생성합니다."""
    print("샘플 데이터를 생성합니다...")
    # Azimuth 범위를 정의
    az_ranges = [
        (0, 45, 'good'),      # 양호한 구간 1
        (45, 60, 'noisy'),     # 노이즈 구간
        (60, 120, 'good'),     # 양호한 구간 2
        (120, 130, 'empty'),   # 데이터 없는 구간
        (130, 180, 'good')     # 양호한 구간 3
    ]
    
    all_points = []
    
    # Elevation은 0도에서 180도까지 1도 간격으로 가정
    el_base = np.arange(0, 180, 1.0)

    for start_az, end_az, kind in az_ranges:
        for az in np.arange(start_az, end_az, 1.0):
            if kind == 'good':
                # el에 따라 dist가 선형적으로 변하는 평면 데이터
                dist = 5000 + 10 * el_base + np.random.normal(0, 20, len(el_base))
            elif kind == 'noisy':
                # dist가 매우 불규칙한 노이즈 데이터
                dist = 5000 + np.random.uniform(-1000, 1000, len(el_base))
            elif kind == 'empty':
                continue # 이 구간은 데이터를 생성하지 않음

            for i in range(len(el_base)):
                all_points.append([dist[i], az, el_base[i]])
    
    df = pd.DataFrame(all_points, columns=['dist', 'az', 'el'])
    print("샘플 데이터 생성 완료!")
    return df

# =============================================================================
# 1. 분석 파라미터
# =============================================================================
ANALYSIS_PARAMS = {
    'ransac_residual_threshold': 500.0,
    'min_inlier_ratio_threshold': 0.85,
}

# =============================================================================
# 2. 편심 보정 좌표 계산 함수 (기존과 동일)
# =============================================================================
def calculate_xyz_with_offset(df):
    scanner_offset = 58.2
    df_calc = df.copy()
    dist = df_calc['dist'].values
    az_rad = np.radians(df_calc['az'].values)
    el_rad = np.radians(df_calc['el'].values + 90)
    x_relative = dist * np.cos(el_rad) * np.sin(az_rad)
    y_relative = dist * np.cos(el_rad) * np.cos(az_rad)
    z_relative = dist * np.sin(el_rad)
    scanner_x = scanner_offset * np.cos(az_rad)
    scanner_y = scanner_offset * np.sin(az_rad)
    df_calc['x'] = + scanner_x + x_relative
    df_calc['y'] = + scanner_y - y_relative
    df_calc['z'] = z_relative
    return df_calc

# =============================================================================
# 3. (수정) 핵심 분석 함수 + 개별 시각화 로직 추가
# =============================================================================
def analyze_scan_by_ransac(scan_df, params, debug_azimuths=None):
    """
    RANSAC 분석을 수행하고, debug_azimuths에 지정된 각도에 대한
    개별 RANSAC 피팅 과정을 시각화합니다.
    """
    if debug_azimuths is None:
        debug_azimuths = []
        
    el_mask = (scan_df['el'] < 90) | (scan_df['el'] > 270)
    df_filtered = scan_df[el_mask]
    if df_filtered.empty: return None, []

    azimuths = df_filtered['az'].unique()
    metrics = []

    for az in sorted(azimuths):
        slice_df = df_filtered[df_filtered['az'] == az]
        if len(slice_df) < 10:
            continue

        X = slice_df['el'].values.reshape(-1, 1)
        y = slice_df['dist'].values

        try:
            ransac = RANSACRegressor(
                residual_threshold=params['ransac_residual_threshold'],
                max_trials=100
            )
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(X)
            
            # ⭐️ 디버깅 시각화 호출 지점 ⭐️
            if az in debug_azimuths:
                visualize_ransac_slice(X, y, ransac, inlier_mask, inlier_ratio, az)

        except ValueError:
            inlier_ratio = 0

        metrics.append({'az': az, 'inlier_ratio': inlier_ratio})

    if not metrics: return None, []
    metrics_df = pd.DataFrame(metrics)

    is_valid = metrics_df['inlier_ratio'] >= params['min_inlier_ratio_threshold']
    valid_azimuths = metrics_df['az'][is_valid].values

    if len(valid_azimuths) == 0: return metrics_df, []

    valid_groups = []
    start_az = valid_azimuths[0]
    for i in range(1, len(valid_azimuths)):
        if (valid_azimuths[i] - valid_azimuths[i-1]) > 2.0:
            valid_groups.append([start_az, valid_azimuths[i-1]])
            start_az = valid_azimuths[i]
    valid_groups.append([start_az, valid_azimuths[-1]])

    return metrics_df, valid_groups

# =============================================================================
# 4. (추가) 개별 Azimuth 슬라이스 시각화 함수
# =============================================================================
def visualize_ransac_slice(X, y, ransac_model, inlier_mask, inlier_ratio, azimuth):
    """
    단일 Azimuth 슬라이스에 대한 RANSAC 결과를 시각화합니다.
    Inlier, Outlier, 그리고 피팅된 직선을 보여줍니다.
    """
    outlier_mask = ~inlier_mask
    
    plt.figure(figsize=(12, 7))
    
    # 모든 점들을 파란색으로 표시
    plt.scatter(X, y, color='dodgerblue', marker='.', label='All Points')
    
    # Inlier(유효점)들을 초록색으로 덧씌워 표시
    plt.scatter(X[inlier_mask], y[inlier_mask], color='limegreen', marker='o', label='Inliers')
    
    # Outlier(무효점)들을 빨간색으로 덧씌워 표시
    plt.scatter(X[outlier_mask], y[outlier_mask], color='crimson', marker='x', label='Outliers')
    
    # RANSAC으로 피팅된 직선 그리기
    line_X = np.arange(X.min(), X.max()).reshape(-1, 1)
    line_y_ransac = ransac_model.predict(line_X)
    plt.plot(line_X, line_y_ransac, color='darkorange', linestyle='-', linewidth=2, label='RANSAC Regressor')
    
    plt.title(f'RANSAC Fit for Azimuth = {azimuth:.2f}° (Inlier Ratio: {inlier_ratio:.2%})', fontsize=16)
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Distance (mm)')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# 5. 시각화 함수 (기존과 동일)
# =============================================================================
def visualize_ransac_analysis(metrics_df, valid_groups, params):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(metrics_df['az'], metrics_df['inlier_ratio'], label='RANSAC Inlier Ratio', color='dodgerblue')
    ax.axhline(y=params['min_inlier_ratio_threshold'], color='r', linestyle='--',
               label=f"Threshold ({params['min_inlier_ratio_threshold']})")
    ax.set_title('RANSAC Inlier Ratio Analysis & Filtering Decision', fontsize=16)
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Inlier Ratio (0.0 to 1.0)')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True)
    for start, end in valid_groups:
        ax.axvspan(start, end, color='green', alpha=0.15, label='Valid Group' if 'Valid Group' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. (수정) 메인 실행부
# =============================================================================
if __name__ == "__main__":
    
    # ⭐️ True로 바꾸면 샘플 데이터로 실행, False로 바꾸면 파일 선택창이 뜹니다.
    USE_SAMPLE_DATA = False

    if USE_SAMPLE_DATA:
        df = generate_sample_data()
    else:
        root = tk.Tk(); root.withdraw()
        filepath = filedialog.askopenfilename(title="분석할 CSV 파일 선택", filetypes=[("CSV Files", "*.csv")])
        if filepath:
            print(f"파일 로딩: {os.path.basename(filepath)}")
            df = pd.read_csv(filepath, header=None, names=['dist', 'az', 'el'])
        else:
            df = None # 파일 선택 안하면 df를 None으로 설정

    if df is not None:
        # ⭐️⭐️ 디버깅하고 싶은 Azimuth 각도를 여기에 추가! ⭐️⭐️
        # '양호한' 구간의 대표값과 '노이즈' 구간의 대표값을 넣어 비교해보세요.
        debug_azimuths_to_show = [30.0, 50.0] 
        
        metrics_df, valid_groups = analyze_scan_by_ransac(df, ANALYSIS_PARAMS, debug_azimuths=debug_azimuths_to_show)

        if metrics_df is not None:
            print("\n[분석 결과]")
            print(f"-> 최종 {len(valid_groups)}개의 유효 그룹 선택됨:")
            for start, end in valid_groups:
                print(f"   - Azimuth 범위: {start:.2f}° ~ {end:.2f}°")

            # 전체 분석 결과 시각화
            visualize_ransac_analysis(metrics_df, valid_groups, ANALYSIS_PARAMS)
            
            # 3D 포인트 클라우드 시각화
            print("\n3D 뷰어에서 유효/비유효 영역을 확인하세요...")
            pcd_df = calculate_xyz_with_offset(df)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_df[['x', 'y', 'z']].values)
            point_colors = np.full((len(df), 3), [0.5, 0.5, 0.5]) # 기본 회색

            az_mask = pd.Series(False, index=df.index)
            for start, end in valid_groups:
                az_mask |= df['az'].between(start, end)
            
            # az_mask와 point_colors의 길이가 다를 경우를 대비한 안전장치
            valid_indices = df.index[az_mask].tolist()
            if len(valid_indices) > 0:
                point_colors[valid_indices] = [0, 1, 0] # 초록색

            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            o3d.visualization.draw_geometries([pcd], window_name="Valid (Green) vs Invalid (Gray) Points")