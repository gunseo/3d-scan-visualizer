import pandas as pd
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import os
import copy
from collections import defaultdict

# =============================================================================
# 섹션 0: 수동 정렬 및 시각화 도우미 (수정 없음)
# =============================================================================
def manual_tweak_registration(source_pcd, target_pcd, T_init, t_step=10.0, r_step_deg=1.0):
    """
    [단순화 버전] Source와 Target을 각각 단색으로 구분하여 수동 미세조정을 수행합니다.
    source_pcd: 움직일 Open3D 포인트 클라우드 객체
    target_pcd: 기준이 되는 Open3D 포인트 클라우드 객체
    """
    # --- ▼▼▼ 색상 로직 단순화 ▼▼▼ ---
    #고정된 색상을 사용합니다.
    tgt = o3d.geometry.PointCloud(target_pcd)
    src0 = o3d.geometry.PointCloud(source_pcd) 
    D = {"mat": np.eye(4)}
    # --- ▲▲▲ 로직 단순화 완료 ▲▲▲ ---

    def rot(axis, deg):
        th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
        if axis == 'x': R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 'y': R = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else: R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        M = np.eye(4); M[:3,:3] = R; return M
    def trans(dx,dy,dz):
        M = np.eye(4); M[:3,3] = [dx,dy,dz]; return M
    def apply(M): D["mat"] = M @ D["mat"]

    def redraw(vis):
        vis.clear_geometries()
        
        # Target 포인트 클라우드는 회색으로 표시
        tgt_show = o3d.geometry.PointCloud(tgt)
        tgt_show.paint_uniform_color([0.7, 0.7, 0.7])
        vis.add_geometry(tgt_show)
        
        # Source 포인트 클라우드는 빨간색으로 표시
        src_show = o3d.geometry.PointCloud(src0)
        src_show.transform(D["mat"] @ T_init)
        src_show.paint_uniform_color([1, 0, 0]) # <- 단색으로 칠하는 부분!
        vis.add_geometry(src_show)
        
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    # 창 제목도 간결하게 변경
    vis.create_window("Manual Tweak (WASDRF, IJKLOU) | Z=Save, Close=Cancel")
    redraw(vis)
    
    # (키 콜백 부분은 기존과 동일)
    vis.register_key_callback(ord('A'), lambda v: (apply(trans(-t_step,0,0)), redraw(v))[1])
    vis.register_key_callback(ord('D'), lambda v: (apply(trans(+t_step,0,0)), redraw(v))[1])
    vis.register_key_callback(ord('W'), lambda v: (apply(trans(0,+t_step,0)), redraw(v))[1])
    vis.register_key_callback(ord('S'), lambda v: (apply(trans(0,-t_step,0)), redraw(v))[1])
    vis.register_key_callback(ord('R'), lambda v: (apply(trans(0,0,+t_step)), redraw(v))[1])
    vis.register_key_callback(ord('F'), lambda v: (apply(trans(0,0,-t_step)), redraw(v))[1])
    vis.register_key_callback(ord('I'), lambda v: (apply(rot('x',-r_step_deg)), redraw(v))[1])
    vis.register_key_callback(ord('K'), lambda v: (apply(rot('x',+r_step_deg)), redraw(v))[1])
    vis.register_key_callback(ord('J'), lambda v: (apply(rot('y',-r_step_deg)), redraw(v))[1])
    vis.register_key_callback(ord('L'), lambda v: (apply(rot('y',+r_step_deg)), redraw(v))[1])
    vis.register_key_callback(ord('U'), lambda v: (apply(rot('z',-r_step_deg)), redraw(v))[1])
    vis.register_key_callback(ord('O'), lambda v: (apply(rot('z',+r_step_deg)), redraw(v))[1])
    vis.register_key_callback(32, lambda v: (D.update(mat=np.eye(4)), redraw(v))[1])
    confirmed = {"ok": False}
    vis.register_key_callback(ord('Z'), lambda v: (confirmed.update(ok=True), v.close())[1])
    
    vis.run(); vis.destroy_window()
    return (D["mat"] @ T_init)

# =============================================================================
# 섹션 1: 유틸리티 함수 (기존 함수들, 수정 없음)
# =============================================================================
def preprocess_pcd(pcd, fpfh_size):
    pcd_down = pcd.voxel_down_sample(fpfh_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_size * 2, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_size * 5, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, fpfh_size):
    distance_threshold = fpfh_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999))
    return result

def refine_registration(source, target, initial_transform, fpfh_size):
    distance_threshold = fpfh_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# =============================================================================
# 섹션 2: 데이터 처리 함수 (⭐️⭐️⭐️ 새로 추가/수정된 부분 ⭐️⭐️⭐️)
# =============================================================================
def load_all_scans_to_dataframe(file_paths):
    """
    모든 CSV 파일을 하나의 Pandas DataFrame으로 통합 로딩합니다.
    이때, 스캐너의 편심 회전 오프셋을 보정하여 좌표를 계산합니다.
    """
    all_dfs = []
    scanner_offset = 58.2  # 편심 회전 오프셋 값 (mm)

    for i, path in enumerate(file_paths):
        try:
            df = pd.read_csv(path, header=None, names=['dist', 'az', 'el'])
            df['scanner_id'] = i  # 어떤 스캐너에서 왔는지 ID 부여

            # az, el 순으로 정렬 (데이터 처리 일관성 유지)
            df_sorted = df.sort_values(by=['az', 'el']).copy()

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
            # z_offset은 0으로 가정

            # 3. 최종 좌표 = 측정 헤드 위치 + 상대 좌표
            df_sorted['x'] = + scanner_x + x_relative
            df_sorted['y'] = + scanner_y - y_relative
            df_sorted['z'] = z_relative
            
            all_dfs.append(df_sorted)
            print(f"-> '{os.path.basename(path)}' 로드 및 편심 보정 완료, 스캐너 ID: {i}")

        except Exception as e:
            print(f"오류: '{path}' 파일 읽기 실패: {e}")
            
    # 모든 DataFrame을 하나로 합침
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def apply_transforms_to_dataframe(df, transforms):
    """계산된 변환 행렬을 DataFrame의 각 포인트에 적용합니다."""
    df_transformed = df.copy()
    for scanner_id, transform in enumerate(transforms):
        if scanner_id == 0: continue
        indices = df_transformed[df_transformed['scanner_id'] == scanner_id].index
        points = df_transformed.loc[indices, ['x', 'y', 'z']].values
        points_h = np.hstack((points, np.ones((len(points), 1))))
        transformed_points_h = (transform @ points_h.T).T
        df_transformed.loc[indices, ['x', 'y', 'z']] = transformed_points_h[:, :3]
    print("-> 모든 포인트에 변환 행렬 적용 완료.")
    return df_transformed

def optimal_voxel_partitioning(df, voxel_size):
    """
    Voxel 내 포인트 개수를 기준으로 Owner Scanner와 Azimuth 정보를 결정합니다.
    (transforms 파라미터는 더 이상 필요하지 않습니다.)
    """
    print("\n[3/4] Voxel 기반 공간 분할 (개선된 방식) 시작...")
    
    # 각 포인트가 어떤 Voxel에 속하는지 계산
    # 이 과정은 법선 벡터 계산이 필요 없으므로 앞으로 이동 가능
    df['voxel_index'] = [tuple(idx) for idx in np.floor(df[['x', 'y', 'z']].values / voxel_size).astype(int)]
    
    # Voxel별로 포인트 인덱스 그룹화
    voxel_to_points = df.groupby('voxel_index').groups
    print(f"-> {len(voxel_to_points)}개의 고유 Voxel 생성됨.")

    results = []
    
    for voxel_index, point_indices in voxel_to_points.items():
        voxel_points_df = df.loc[point_indices]
        
        # Voxel 내에 포인트가 없는 경우 건너뛰기
        if voxel_points_df.empty:
            continue
            
        # Voxel 내에서 스캐너 ID별로 포인트 개수를 계산
        scanner_counts = voxel_points_df['scanner_id'].value_counts()
        
        # 포인트가 가장 많은 스캐너를 Owner로 결정
        best_scanner = scanner_counts.idxmax()
        
        # Owner 스캐너가 생성한 포인트들의 Azimuth 정보 수집
        owner_points_df = voxel_points_df[voxel_points_df['scanner_id'] == best_scanner]
        azimuths = sorted(owner_points_df['az'].unique().tolist())
        
        voxel_center = (np.array(voxel_index) + 0.5) * voxel_size
        results.append({
            'voxel_center_x': voxel_center[0], 
            'voxel_center_y': voxel_center[1], 
            'voxel_center_z': voxel_center[2],
            'owner_scanner_id': best_scanner, 
            'contributing_azimuths': str(azimuths)
        })
        
    print(f"-> {len(results)}개 Voxel의 주인 결정 완료.")
    return pd.DataFrame(results)
class StatisticalOutlierRemover:
    """
    Open3D의 통계적 이상치 제거(statistical outlier removal) 기능을
    캡슐화한 클래스입니다.

    이 클래스를 사용하면 필터링 파라미터를 객체에 저장해두고
    여러 포인트 클라우드에 일관되게 적용할 수 있습니다.
    """
    def __init__(self, nb_neighbors, std_ratio):
        """
        필터 객체를 초기화하고 파라미터를 설정합니다.

        Args:
            nb_neighbors (int): 각 점의 평균 거리를 계산할 때 고려할 이웃 점의 수.
            std_ratio (float): 이상치로 간주할 표준 편차의 배수.
                               값이 작을수록 더 많은 점을 이상치로 제거합니다.
        """
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio
        print(f"필터 생성됨: 이웃 수={self.nb_neighbors}, 표준편차 비율={self.std_ratio}")

    def filter(self, pcd):
        """
        입력된 포인트 클라우드에 대해 통계적 이상치 제거를 수행합니다.

        Args:
            pcd (o3d.geometry.PointCloud): 필터링을 적용할 포인트 클라우드 객체.

        Returns:
            tuple: (filtered_pcd, inlier_indices)
                - filtered_pcd (o3d.geometry.PointCloud): 이상치가 제거된 포인트 클라우드.
                - inlier_indices (o3d.utility.IntVector): 원본 포인트 클라우드에서 살아남은 점(inlier)들의 인덱스.
        """
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError("입력값은 open3d.geometry.PointCloud 객체여야 합니다.")

        print("이상치 제거 필터링 시작...")
        # remove_statistical_outlier 함수는 (포인트클라우드, 인덱스) 튜플을 반환합니다.
        filtered_pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.nb_neighbors,
            std_ratio=self.std_ratio
        )
        print(f"필터링 완료: 원본 {len(pcd.points)}개 -> 제거 후 {len(filtered_pcd.points)}개")
        return filtered_pcd, ind
# =============================================================================
# 메인 실행부 (⭐️⭐️⭐️ 이상치 제거 단계 추가 ⭐️⭐️⭐️)
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    print("분석할 CSV 파일들을 순서대로 선택하세요 (2개 이상)")
    file_paths = filedialog.askopenfilenames(title="CSV 파일 선택 (기준 파일 먼저)", filetypes=[("CSV Files", "*.csv")])

    if len(file_paths) < 2:
        print("파일이 2개 미만 선택되어 프로그램을 종료합니다.")
    else:
        # --- 1단계: 모든 CSV를 하나의 DataFrame으로 로딩 ---
        print("\n[1/5] CSV 파일 로딩 및 통합...")
        master_df = load_all_scans_to_dataframe(file_paths)
        if master_df.empty: exit()

        # --- ⭐️ 2단계: 각 스캔에 대해 이상치 제거 수행 (새로 추가된 단계) ⭐️ ---
        print("\n[2/5] 통계적 이상치 제거 시작...")
        # 1. 이상치 제거 필터 객체 생성 (파라미터는 데이터에 맞게 조정)
        sor_filter = StatisticalOutlierRemover(nb_neighbors=100, std_ratio=2.0)
        
        cleaned_dfs = []
        for i in range(len(file_paths)):
            scan_df = master_df[master_df['scanner_id'] == i]
            
            # DataFrame을 Open3D 포인트 클라우드로 변환
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(scan_df[['x', 'y', 'z']].values)
            
            # 필터 적용
            print(f"-> 스캐너 ID {i} 필터링 중...")
            filtered_pcd, inlier_indices = sor_filter.filter(pcd)
            
            # 살아남은 인덱스(inlier_indices)를 사용해 원본 DataFrame 필터링
            # np.asarray를 통해 o3d.utility.IntVector를 NumPy 배열로 변환
            cleaned_scan_df = scan_df.iloc[np.asarray(inlier_indices)]
            cleaned_dfs.append(cleaned_scan_df)

        # 깨끗해진 DataFrame들을 다시 하나로 합침
        master_df_cleaned = pd.concat(cleaned_dfs, ignore_index=True)
        print("✅ 모든 스캔의 이상치 제거 완료.")


        # --- 3단계: 자동 정합을 위한 변환 행렬 계산 (기존 2단계) ---
        # ⭐️ 이제부터는 깨끗해진 master_df_cleaned를 사용합니다.
        print("\n[3/5] 정합 변환 행렬 계산 시작...")
        voxel_size = 300.0
        fpfh_size= 30
        
        pcds_for_reg = [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            master_df_cleaned[master_df_cleaned['scanner_id'] == i][['x', 'y', 'z']].values
        )) for i in range(len(file_paths))]

        final_transforms = [np.identity(4)]
        target_pcd_merged = copy.deepcopy(pcds_for_reg[0])

        for i in range(1, len(pcds_for_reg)):
            source_pcd = pcds_for_reg[i]
            print(f"\n-> 스캔 {i} ('{os.path.basename(file_paths[i])}') 정합 중...")
            
            source_down, source_fpfh = preprocess_pcd(source_pcd, fpfh_size)
            target_down, target_fpfh = preprocess_pcd(target_pcd_merged, fpfh_size)

            ransac_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, fpfh_size)
            
            print("\n👀 RANSAC 결과 기반 수동 미세조정: 키보드로 조정 후 'Z'로 확정!")
            
            # ⭐️ current_source_df도 깨끗한 데이터프레임에서 가져옵니다.
            current_source_df = master_df_cleaned[master_df_cleaned['scanner_id'] == i]
            
            T_tweaked = manual_tweak_registration( # 함수 이름 변경
                source_pcd, 
                target_pcd_merged, 
                ransac_result.transformation
            )
            
            target_pcd_merged.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_size * 2, max_nn=30))
            icp_result = refine_registration(source_pcd, target_pcd_merged, T_tweaked, fpfh_size)
            
            final_transforms.append(icp_result.transformation)
            
            source_pcd.transform(icp_result.transformation)
            target_pcd_merged += source_pcd
        print("✅ 모든 변환 행렬 계산 완료.")

        # --- 3.5단계 -> 4단계: 변환 행렬 적용 ---
        print("\n[4/5] 변환 행렬 적용...")
        # ⭐️ 변환 행렬은 깨끗한 데이터프레임에 적용합니다.
        transformed_df = apply_transforms_to_dataframe(master_df_cleaned, final_transforms)

        # --- 4단계 -> 5단계: Voxel 기반 공간 분할 및 저장/시각화 ---
        print("\n[5/5] Voxel 기반 공간 분할, 저장 및 시각화...")
        results_df = optimal_voxel_partitioning(transformed_df, voxel_size)
        
        # (이하 저장 및 시각화 코드는 기존과 동일)
        output_dir = "final_voxel_results"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, "voxel_ownership_with_azimuth.csv")
        results_df.to_csv(output_filename, index=False)
        print(f"-> 최종 Voxel 결과 저장 완료: '{output_filename}'")

        pcd_final = o3d.geometry.PointCloud()
        pcd_final.points = o3d.utility.Vector3dVector(transformed_df[['x', 'y', 'z']].values)
        
        voxel_owner_map = results_df.set_index(results_df.apply(
            lambda row: tuple(np.floor(row[['voxel_center_x', 'voxel_center_y', 'voxel_center_z']].values / voxel_size).astype(int)), axis=1
        ))['owner_scanner_id'].to_dict()

        point_colors = np.zeros_like(transformed_df[['x', 'y', 'z']].values)
        partition_colors = np.random.rand(len(file_paths), 3)
        
        owner_ids = transformed_df['voxel_index'].map(voxel_owner_map).fillna(-1).astype(int)
        valid_indices = owner_ids != -1
        point_colors[valid_indices] = partition_colors[owner_ids[valid_indices]]
        
        pcd_final.colors = o3d.utility.Vector3dVector(point_colors)
        
        o3d.visualization.draw_geometries([pcd_final], window_name="Voxel 기반 공간 분할 결과")
        
        print("\n✨ 모든 작업이 완료되었습니다.")