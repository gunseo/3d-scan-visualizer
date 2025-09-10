# -*- coding: utf-8 -*-
"""
3D 스캔 데이터(CSV) 클러스터링 최종 스크립트 (엄격한 라운드 기반 버전)
- 초기 와이어프레임 시각화 기능 추가
"""

import pandas as pd
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm
import matplotlib.cm as cm

# --------------------------------------------------------------------------
# 섹션 1 & 2: 기본/데이터 준비 함수 (변경 없음)
# --------------------------------------------------------------------------
def get_triangle_plane(p1, p2, p3):
    v1 = p2 - p1; v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9: return None, None
    normal /= norm_len
    d = -np.dot(normal, p1)
    return normal, d

def fit_plane_pca(points):
    if len(points) < 3: return None, None
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    d = -np.dot(normal, centroid)
    return normal, d

import pandas as pd
import numpy as np

def load_and_filter_points_from_csv(file_path,min_distance):
    """
    CSV를 로드하고 az/el 정렬 후 3D 포인트를 계산합니다.
    그 다음, 특정 XYZ 좌표 조건에 따라 포인트를 필터링합니다.
    (z<0 이면서 x^2+y^2 < 35^2 인 점 제거)
    """
    # 1. CSV 로드 및 컬럼 이름 지정
    df = pd.read_csv(file_path, header=None)
    df.columns = ['dist', 'az', 'el']

    # 2. 'az'와 'el' 기준으로 정렬
    sorted_df = df.sort_values(by=['az', 'el'])

     # 3. 정렬된 데이터를 기반으로 3D 좌표 계산
    dist = sorted_df['dist'].values
    az_rad = np.radians(sorted_df['az'].values)
    el_rad = np.radians(sorted_df['el'].values + 90)

    # 먼저, 움직이는 스캐너로부터의 '상대' 좌표를 계산합니다.
    # (이것이 기존 코드의 계산 방식이었습니다)
    x_relative = dist * np.cos(el_rad) * np.cos(az_rad)
    y_relative = dist * np.cos(el_rad) * np.sin(az_rad)
    z_relative = dist * np.sin(el_rad)

    # --- ▼▼▼ 편심 회전(Eccentric Rotation) 오프셋 보정 ▼▼▼ ---
    scanner_offset = 58.2

    # 각 포인트가 측정될 당시의 '스캐너 자체의 위치'를 계산합니다.
    # az_rad는 위에서 이미 계산해두었습니다.
    scanner_x = scanner_offset * np.cos(az_rad)
    scanner_y = scanner_offset * np.sin(az_rad)
    # z_offset은 0이라고 가정합니다 (수평 회전)

    # 최종 실제 좌표 = 스캐너 위치 + 상대 측정 좌표
    x = scanner_x + x_relative
    y = scanner_y + y_relative
    z = z_relative # z 좌표는 변하지 않습니다.
    # --- ▲▲▲ 보정 완료 ▲▲▲ ---

    # 4. 필터링을 위해 계산된 x, y, z 좌표를 DataFrame에 임시로 추가
    sorted_df['x'] = x
    sorted_df['y'] = y
    sorted_df['z'] = z
    
    # --- ▼▼▼ 요청하신 필터링 로직 ▼▼▼ ---

    # 5. 필터링 조건 정의
    # 제거할 조건: (z가 0보다 작고) AND (x^2 + y^2이 35^2보다 작다)
    initial_count = len(sorted_df)
    condition_to_remove = (sorted_df['z'] < 200) & \
                          (sorted_df['x']**2 + sorted_df['y']**2 < min_distance**2)
    
    # 6. 위 조건을 만족하지 않는(~) 행들만 남겨서 필터링 수행
    filtered_df = sorted_df[~condition_to_remove].copy()
    
    print(f"XYZ 필터링: {initial_count}개 -> {len(filtered_df)}개 (z<0 & x²+y²<35² 조건 제외)")

    # 7. 필터링된 DataFrame에서 최종 points 배열 생성
    #    (x, y, z 컬럼만 선택하여 Numpy 배열로 변환)
    final_points = filtered_df[['x', 'y', 'z']].to_numpy()
    
    # 8. 필터링된 결과 반환 (이때 df는 x,y,z 컬럼을 포함하고 있음)
    return final_points, filtered_df

def build_triangles_with_origin(df, points, max_edge_length):
    """
    DataFrame과 3D 포인트를 기반으로 삼각형을 생성합니다.
    이때, 한 변의 길이가 max_edge_length를 초과하는 비정상적인 삼각형은 제외합니다.
    """
    grid = {(int(row.iloc[1]), int(row.iloc[2])): idx for idx, row in df.iterrows()}
    triangles, triangle_origins = [], []
    az_steps = sorted(list(set(df.iloc[:, 1].astype(int))))
    el_steps = sorted(list(set(df.iloc[:, 2].astype(int))))
    
    # --- ▼▼▼ 수정된 부분 (1/2): 거리 비교를 위한 제곱값 계산 ▼▼▼ ---
    # 제곱근 계산을 피하기 위해, 길이의 제곱을 기준으로 비교합니다. (계산 속도 향상)
    max_len_sq = max_edge_length ** 2
    # --- ▲▲▲ 수정 완료 ▲▲▲ ---

    for i in range(len(az_steps) - 1): 
        az = az_steps[i]
        az_next = az_steps[i + 1]
        for j in range(len(el_steps) - 1): # el_steps도 마지막 연결을 하지 않도록 수정
            el = el_steps[j]
            el_next = el_steps[j + 1]
            
            p1_idx, p2_idx = grid.get((az, el)), grid.get((az_next, el))
            p3_idx, p4_idx = grid.get((az, el_next)), grid.get((az_next, el_next))
            
            # 첫 번째 삼각형 (p1, p2, p3) 유효성 검사
            if p1_idx is not None and p2_idx is not None and p3_idx is not None:
                # --- ▼▼▼ 수정된 부분 (2/2): 변 길이 체크 로직 추가 ▼▼▼ ---
                p1, p2, p3 = points[p1_idx], points[p2_idx], points[p3_idx]
                if (np.sum((p1-p2)**2) < max_len_sq and 
                    np.sum((p2-p3)**2) < max_len_sq and 
                    np.sum((p3-p1)**2) < max_len_sq):
                    triangles.append((p1_idx, p2_idx, p3_idx))
                    triangle_origins.append(i)
                # --- ▲▲▲ 수정 완료 ▲▲▲ ---

            # 두 번째 삼각형 (p3, p2, p4) 유효성 검사
            if p3_idx is not None and p2_idx is not None and p4_idx is not None:
                # --- ▼▼▼ 수정된 부분 (2/2): 변 길이 체크 로직 추가 ▼▼▼ ---
                p3, p2, p4 = points[p3_idx], points[p2_idx], points[p4_idx]
                if (np.sum((p3-p2)**2) < max_len_sq and 
                    np.sum((p2-p4)**2) < max_len_sq and 
                    np.sum((p4-p3)**2) < max_len_sq):
                    triangles.append((p3_idx, p2_idx, p4_idx))
                    triangle_origins.append(i)
                # --- ▲▲▲ 수정 완료 ▲▲▲ ---

    return triangles, triangle_origins

def prepare_mesh_data(points, triangles):
    edge_to_triangles = defaultdict(list)
    for i, (p1, p2, p3) in enumerate(triangles):
        edges = [tuple(sorted(e)) for e in [(p1, p2), (p2, p3), (p3, p1)]]
        for edge in edges: edge_to_triangles[edge].append(i)
    adjacencies = [[] for _ in range(len(triangles))]
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            t1, t2 = tris; adjacencies[t1].append(t2); adjacencies[t2].append(t1)
    return adjacencies

# --------------------------------------------------------------------------
# 섹션 3: 핵심 클러스터링 알고리즘 (변경 없음)
# --------------------------------------------------------------------------
def professional_graph_merge(points, triangles, initial_clusters, initial_graph, angle_deg_tolerance, show_progress=True):
    """(최종 전문가 버전 + tqdm 최적화) 그래프 엣지 축약 기반의 고성능 클러스터링 함수."""
    if not initial_clusters: return []

    cos_thresh = math.cos(math.radians(angle_deg_tolerance))
    clusters = initial_clusters.copy()
    graph = {k: v.copy() for k, v in initial_graph.items()}
    next_cluster_id = max(clusters.keys()) + 1 if clusters else 0

    pass_num = 0
    while True:
        pass_num += 1
        num_clusters = len(clusters)
        if num_clusters <= 1: break

        if show_progress:
            print(f"\n--- Pass {pass_num} 시작 (현재 클러스터 개수: {num_clusters}) ---")

        cluster_ids = sorted(list(clusters.keys()))
        cluster_planes = {cid: fit_plane_pca(points[list(set(p for t in clusters[cid] for p in triangles[t]))]) for cid in cluster_ids}

        merged_in_pass = set()
        iterator = tqdm(cluster_ids, desc=f"   - Pass {pass_num} 진행 중") if show_progress else cluster_ids

        for cid in iterator:
            if cid in merged_in_pass or cluster_planes[cid][0] is None:
                continue

            for neighbor_id in list(graph.get(cid, set())):
                if neighbor_id in merged_in_pass or cluster_planes.get(neighbor_id, (None,))[0] is None:
                    continue

                dot_product = abs(np.dot(cluster_planes[cid][0], cluster_planes[neighbor_id][0]))
                if dot_product >= cos_thresh:
                    new_id = next_cluster_id; next_cluster_id += 1
                    clusters[new_id] = clusters[cid] + clusters[neighbor_id]
                    graph[new_id] = (graph[cid] | graph[neighbor_id]) - {cid, neighbor_id}
                    for n_id in graph[new_id]:
                        graph[n_id].discard(cid)
                        graph[n_id].discard(neighbor_id)
                        graph[n_id].add(new_id)
                    del clusters[cid]; del clusters[neighbor_id]
                    del graph[cid]; del graph[neighbor_id]
                    merged_in_pass.add(cid); merged_in_pass.add(neighbor_id)
                    break

        if not merged_in_pass:
            if show_progress:
                print("\n더 이상 병합할 클러스터가 없어 최종 완료되었습니다.")
            break

    return list(clusters.values())

# --------------------------------------------------------------------------
# 섹션 4: 시각화 함수 (와이어프레임 함수 추가)
# --------------------------------------------------------------------------
def visualize_without_light(mesh):
    vis = o3d.visualization.Visualizer(); vis.create_window(); vis.add_geometry(mesh)
    opt = vis.get_render_option(); opt.light_on = False; opt.mesh_show_back_face = True; vis.run(); vis.destroy_window()
    

def visualize_edges_only(points, triangles, show_points=True):
    """
    삼각형의 모서리(유일한 에지)만 선으로 시각화.
    필요하면 점도 함께 찍어줍니다(포인트 클라우드).
    """

    # 유일한 에지 집합 생성
    edge_set = set()
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edge_set.add(tuple(sorted((u, v))))

    lines = np.array(list(edge_set), dtype=np.int32)

    # LineSet 구성
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    # (선 색: 검정) 원하면 색 바꿔도 됨
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.0, 0.0, 0.0]]), (len(lines), 1))
    )

    # 시각화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Edges-Only (초기 확인용)")
    vis.add_geometry(line_set)
    if pcd is not None:
        vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.light_on = False          # 조명 끔 (선/점만 또렷)
    opt.point_size = 3.0          # 점 크기
    try:
        opt.line_width = 1.5      # 일부 버전에서만 지원
    except Exception:
        pass

    vis.run()
    vis.destroy_window()

def visualize_points_by_distance(points):
    """
    포인트 클라우드를 원점(0,0,0)으로부터의 거리에 따라 색을 다르게 하여 시각화합니다.
    """
    # 1. 각 점과 원점 사이의 유클리드 거리 계산
    distances = np.linalg.norm(points, axis=1)

    # 2. 거리를 0과 1 사이의 값으로 정규화 (색상 맵에 매핑하기 위함)
    # 거리가 모두 동일할 경우 0으로 나눔 에러 방지
    min_dist, max_dist = np.min(distances), np.max(distances)
    if min_dist == max_dist:
        norm_distances = np.zeros_like(distances)
    else:
        norm_distances = (distances - min_dist) / (max_dist - min_dist)

    # 3. 정규화된 거리에 viridis 색상 맵을 적용하여 RGB 색상 생성
    # matplotlib의 colormap을 사용합니다. 다른 맵(예: 'jet', 'plasma')을 사용해도 좋습니다.
    colors = cm.viridis(norm_distances)[:, :3]  # RGBA에서 RGB 부분만 사용

    # 4. PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 시각화
    # 간단한 시각화를 위해 draw_geometries 사용
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="거리에 따라 색상이 지정된 포인트 클라우드"
    )
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


def visualize_wireframe_mesh(points, triangles):
    """주어진 점과 삼각형으로 구성된 메쉬의 모서리만 보이도록 시각화합니다."""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(np.array(triangles))
    )
    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Initial Mesh Wireframe View")
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.light_on = False
    opt.mesh_show_wireframe = True  # ✨ 와이어프레임 모드 활성화
    opt.mesh_show_back_face = True  # 뒷면도 보이도록 설정

    vis.run()
    vis.destroy_window()

def visualize_clusters(points, triangles, clusters):
    # 각 클러스터에 랜덤 색상 할당 (삼각형 인덱스 기준)
    tri_to_color = {}
    for cluster in clusters:
        color = np.random.rand(3)
        for tri_idx in cluster:
            tri_to_color[tri_idx] = color

    # 시각화를 위한 메쉬 재생성
    new_points, new_triangles, new_vertex_colors = [], [], []
    vertex_counter = 0
    for i, tri_indices in enumerate(triangles):
        color = tri_to_color.get(i, [0.5, 0.5, 0.5]) # 클러스터에 없으면 회색
        p1, p2, p3 = points[tri_indices[0]], points[tri_indices[1]], points[tri_indices[2]]
        new_points.extend([p1, p2, p3])
        new_triangles.append([vertex_counter, vertex_counter + 1, vertex_counter + 2]); vertex_counter += 3
        new_vertex_colors.extend([color, color, color])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(new_points)), o3d.utility.Vector3iVector(np.array(new_triangles)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_vertex_colors)); mesh.compute_vertex_normals()
    visualize_without_light(mesh)

def downsample_groups_fps(df):
    """
    DataFrame을 'az' 그룹별로 나누고, Farthest Point Sampling을 사용해
    각 그룹의 포인트 수를 최소 그룹 크기에 맞춰 균일하게 만듭니다.

    Args:
        df (pd.DataFrame): 이상치 제거가 완료된 데이터프레임. 
                           ['dist', 'az', 'el', 'x', 'y', 'z'] 컬럼을 포함해야 함.

    Returns:
        tuple: (downsampled_df, downsampled_points)
            - downsampled_df (pd.DataFrame): 다운샘플링이 완료된 데이터프레임.
            - downsampled_points (np.ndarray): 다운샘플링된 포인트의 [x, y, z] 좌표 배열.
    """
    print("\n[단계] Farthest Point Sampling으로 공간 균일화를 시작합니다...")

    # 'az' 컬럼을 기준으로 그룹화하고 각 그룹의 크기를 계산
    az_groups = df.groupby(df.columns[1]) 
    group_sizes = az_groups.size()

    if group_sizes.empty:
        print(" - 데이터가 없어 다운샘플링을 건너뜁니다.")
        # 데이터가 없을 경우, 입력받은 그대로 반환
        return df, df[['x', 'y', 'z']].to_numpy()

    # 모든 그룹 중에서 가장 작은 그룹의 크기를 찾음
    min_group_size = group_sizes.min()
    print(f" - 모든 AZ 그룹을 최소 포인트 수인 {min_group_size}개로 통일합니다.")

    # 각 그룹을 순회하며 FPS를 적용
    processed_dfs = []
    for name, group_df in tqdm(az_groups, desc=" - FPS 다운샘플링 진행 중"):
        if len(group_df) == min_group_size:
            processed_dfs.append(group_df)
            continue
        
        pcd_group = o3d.geometry.PointCloud()
        pcd_group.points = o3d.utility.Vector3dVector(group_df[['x', 'y', 'z']].to_numpy())
        
        pcd_downsampled = pcd_group.farthest_point_down_sample(min_group_size)
        
        # 다운샘플링된 포인트와 가장 가까운 원본 포인트를 찾아 인덱스 매칭
        kdtree = o3d.geometry.KDTreeFlann(pcd_group)
        original_indices = [kdtree.search_knn_vector_3d(point, 1)[1][0] for point in pcd_downsampled.points]
        
        df_sampled = group_df.iloc[original_indices]
        processed_dfs.append(df_sampled)

    # 처리된 데이터프레임들을 하나로 다시 합치고 정렬
    df_downsampled = pd.concat(processed_dfs)
    df_final = df_downsampled.sort_values(by=[df.columns[1], df.columns[2]]).reset_index(drop=True)
    points_final = df_final[['x', 'y', 'z']].to_numpy()
    
    print(f"다운샘플링 완료: 총 {len(points_final)}개 포인트")
    
    return df_final, points_final
    

# --------------------------------------------------------------------------
# 섹션 5: 메인 실행부 (초기 시각화 로직 적용)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 섹션 5: 메인 실행부 (모든 개선 사항 적용)
# --------------------------------------------------------------------------
def main():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])
    if not file_path: 
        print("파일이 선택되지 않았습니다.")
        return

    # 1. CSV 파일에서 포인트와 DataFrame 로드 (편심 회전 오차 보정 포함)
    points, df = load_and_filter_points_from_csv(file_path, min_distance=400)
    print(f"초기 데이터 로드 완료: {len(points)}개 포인트")

    # --- 1단계: 이상치 제거 ---
    # 로드한 points(Numpy 배열)를 Open3D 포인트 클라우드로 변환
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # StatisticalOutlierRemover 필터 객체 생성 (파라미터는 데이터에 맞게 조절)
    sor_filter = StatisticalOutlierRemover(nb_neighbors=500, std_ratio=5)

    # 필터 실행하여 이상치 제거
    filtered_pcd, inlier_indices = sor_filter.filter(pcd)

    # 필터링 결과를 원래 변수들에 다시 반영
    points = np.asarray(filtered_pcd.points)
    df = df.iloc[inlier_indices]
    df = df.reset_index(drop=True)
    
    #print(f"이상치 제거 후: {len(points)}개 포인트")
    print("\n[초기 확인 단계]")
    # --- 2단계: 공간 분포를 고려한 다운샘플링 (FPS) ---
    print("\n[신규 단계] Farthest Point Sampling으로 공간 균일화를 시작합니다...")
    df, points = downsample_groups_fps(df)

    # --- 3단계: 초기 확인 및 메쉬 생성 ---
    print("\n[초기 확인 단계]")
    visualize_points_by_distance(points)
    
    # 균일화된 df와 points를 가지고 삼각형 생성 ('열린 스캔'용으로 수정된 함수 사용)
    triangles, triangle_origins = build_triangles_with_origin(df, points, max_edge_length=200.0)
    visualize_wireframe_mesh(points, triangles)

    # --- 4단계: 클러스터링을 위한 그래프 생성 ---
    print("\n최초 인접성 그래프를 생성합니다 (전체 과정 중 한 번만 실행됩니다)...")
    edge_to_triangles = defaultdict(list)
    for i, tri in enumerate(tqdm(triangles, desc=" - 그래프 생성")):
        edges = [tuple(sorted(e)) for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]]
        for edge in edges: edge_to_triangles[edge].append(i)

    adjacency_graph = defaultdict(set)
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            t1, t2 = tris
            adjacency_graph[t1].add(t2)
            adjacency_graph[t2].add(t1)

    print(f"총 {len(points)}개의 점과 {len(triangles)}개의 삼각형, {len(adjacency_graph)}개의 노드를 가진 그래프 생성 완료.")

    # --- 5단계: 1차 클러스터링 ('같은 출신' 내에서 병합) ---
    print("\n[1단계] '같은 출신' 그룹 내에서 병합을 시작합니다...")
    origin_groups = defaultdict(list)
    for i, origin_id in enumerate(triangle_origins): origin_groups[origin_id].append(i)

    clusters_from_stage1 = []
    for origin_id, tri_indices in tqdm(origin_groups.items(), desc="1단계 (출신 그룹별) 진행 중"):
        if not tri_indices: continue
        group_clusters = {idx: [idx] for idx in tri_indices}
        group_graph = {idx: {n for n in adjacency_graph.get(idx, set()) if n in group_clusters} for idx in tri_indices}
        merged = professional_graph_merge(points, triangles, group_clusters, group_graph, 10, show_progress=False)
        clusters_from_stage1.extend(merged)

    print(f"\n1단계 결과, 총 {len(clusters_from_stage1)}개의 초기 클러스터가 생성되었습니다.")

    # --- 6단계: 2차 클러스터링 (전체 클러스터 대상 최종 병합) ---
    print("\n[2단계] 초기 클러스터들을 대상으로 최종 병합을 시작합니다...")

    s2_clusters = {i: c for i, c in enumerate(clusters_from_stage1)}
    tri_to_cid_map = {tri_idx: cid for cid, cluster in s2_clusters.items() for tri_idx in cluster}

    print("  - 2단계용 인접 그래프를 생성합니다...")
    s2_graph = defaultdict(set)
    for cid, cluster in tqdm(s2_clusters.items(), desc="  - 2단계 그래프 생성 중"):
        for tri_idx in cluster:
            for neighbor_tri in adjacency_graph.get(tri_idx, set()):
                neighbor_cid = tri_to_cid_map.get(neighbor_tri)
                if neighbor_cid is not None and neighbor_cid != cid:
                    s2_graph[cid].add(neighbor_cid)

    final_clusters = professional_graph_merge(points, triangles, s2_clusters, s2_graph, 10, show_progress=True)
    print(f"\n최종 클러스터 {len(final_clusters)}개가 생성되었습니다.")

    # --- 7단계: 최종 시각화 ---
    if final_clusters:
        visualize_clusters(points, triangles, final_clusters)

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("'tqdm' 라이브러리가 필요합니다. 'pip install tqdm' 명령어로 설치해주세요.")
        exit()
    try:
        import open3d
    except ImportError:
        print("'open3d' 라이브러리가 필요합니다. 'pip install open3d' 명령어로 설치해주세요.")
        exit()
    main()