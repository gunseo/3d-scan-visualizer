# -*- coding: utf-8 -*-
"""
3D 스캔 데이터(CSV)를 불러와 평면을 감지하고 클러스터링하는 스크립트.

작동 순서:
1. CSV 파일을 읽어 3D 포인트 클라우드를 생성합니다.
2. 전체 삼각형 메쉬를 구성하되, 각 삼각형의 '출신 슬라이스' 정보를 기록합니다.
3. 1단계: '같은 출신'의 삼각형들끼리만 먼저 클러스터링하여 '초기 클러스터'를 생성합니다.
4. 2단계: 생성된 초기 클러스터들을 대상으로, 더 정교한 기준으로 '최종 병합'을 수행합니다.
5. 최종 클러스터링 결과를 Open3D로 시각화합니다.
"""

import pandas as pd
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import open3d as o3d
from collections import defaultdict

# --------------------------------------------------------------------------
# 헬퍼 함수 (수정 없음)
# --------------------------------------------------------------------------
def get_triangle_plane(p1, p2, p3):
    v1 = p2 - p1; v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9: return None, None
    normal /= norm_len
    d = -np.dot(normal, p1)
    return normal, d

def are_coplanar(plane1, plane2, angle_deg_tolerance):
    if plane1 is None or plane2 is None or plane1[0] is None or plane2[0] is None: return False
    normal1, _ = plane1
    normal2, _ = plane2
    dot_product = abs(np.dot(normal1, normal2))
    min_dot = math.cos(math.radians(angle_deg_tolerance))
    return dot_product >= min_dot

def fit_plane_pca(points):
    if len(points) < 3: return None, None
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    d = -np.dot(normal, centroid)
    return normal, d

# --------------------------------------------------------------------------
# ✨✨ 요청하신 워크플로우에 맞춘 핵심 함수들 ✨✨
# --------------------------------------------------------------------------

def build_triangles_with_origin(df):
    """
    (✨ 수정됨)
    전체 삼각형과 함께, 각 삼각형이 어떤 'Azimuth 스포크'에서 생성되었는지
    '출신 정보(origin)'를 함께 반환합니다.
    """
    grid = { (int(row.iloc[1]), int(row.iloc[2])): idx for idx, row in df.iterrows() }
    triangles = []
    triangle_origins = []
    
    az_steps = sorted(list(set(df.iloc[:, 1].astype(int))))
    el_steps = sorted(list(set(df.iloc[:, 2].astype(int))))
    
    # Azimuth를 기준으로 순회하여 '세로줄'을 같은 출신으로 묶습니다.
    for i, az in enumerate(az_steps):
        az_next = az_steps[(i + 1) % len(az_steps)]
        for j, el in enumerate(el_steps):
            el_next = el_steps[(j + 1) % len(el_steps)]
            p1, p2 = grid.get((az, el)), grid.get((az_next, el))
            p3, p4 = grid.get((az, el_next)), grid.get((az_next, el_next))
            if p1 is not None and p2 is not None and p3 is not None:
                triangles.append((p1, p2, p3))
                triangle_origins.append(i) # 출신 정보를 Azimuth 인덱스(i)로 저장
            if p3 is not None and p2 is not None and p4 is not None:
                triangles.append((p3, p2, p4))
                triangle_origins.append(i) # 출신 정보를 Azimuth 인덱스(i)로 저장
                
    return triangles, triangle_origins

def prepare_mesh_data(points, triangles):
    """(기존과 동일) 전체 삼각형 목록에 대한 평면 정보와 인접성 그래프를 미리 계산합니다."""
    triangle_planes = []
    for tri_indices in triangles:
        p1, p2, p3 = points[tri_indices[0]], points[tri_indices[1]], points[tri_indices[2]]
        normal, d = get_triangle_plane(p1, p2, p3)
        triangle_planes.append((normal, d) if normal is not None else None)
    edge_to_triangles = defaultdict(list)
    for i, tri_indices in enumerate(triangles):
        edges = [tuple(sorted(e)) for e in [(tri_indices[0], tri_indices[1]), (tri_indices[1], tri_indices[2]), (tri_indices[2], tri_indices[0])]]
        for edge in edges:
            edge_to_triangles[edge].append(i)
    adjacencies = [[] for _ in range(len(triangles))]
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            adjacencies[tris[0]].append(tris[1])
            adjacencies[tris[1]].append(tris[0])
    return adjacencies, triangle_planes

def cluster_by_origin(triangles, adjacencies, triangle_planes, triangle_origins,
                      seed_angle_deg, local_angle_deg):
    """
    (✨✨ 결정적 오류 수정)
    1단계: '같은 출신'끼리 클러스터링하되, 평면 계산이 안 되는 삼각형도
    '단일 클러스터'로 취급하여 절대 누락시키지 않습니다.
    """
    visited = [False] * len(triangles)
    clusters = []
    
    # 1. 평면 계산이 가능한 '정상' 삼각형들 먼저 클러스터링
    for i in range(len(triangles)):
        if visited[i] or triangle_planes[i] is None: continue
        
        seed_plane = triangle_planes[i]
        origin_idx = triangle_origins[i]
        
        stack = [i]
        visited[i] = True
        current_cluster = []
        
        while stack:
            curr = stack.pop()
            current_cluster.append(curr)
            curr_plane = triangle_planes[curr]
            if curr_plane is None: continue

            for nb in adjacencies[curr]:
                if visited[nb] or triangle_planes[nb] is None: continue
                
                if triangle_origins[nb] != origin_idx:
                    continue
                
                cond_seed = are_coplanar(seed_plane, triangle_planes[nb], angle_deg_tolerance=seed_angle_deg)
                cond_local = are_coplanar(curr_plane, triangle_planes[nb], angle_deg_tolerance=local_angle_deg)
                if cond_seed and cond_local:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(current_cluster)

    return clusters

def refine_and_merge_clusters(points, triangles, initial_clusters, adjacencies, 
                              merge_seed_angle_deg, merge_local_angle_deg):
    """
    (✨ 수정됨) 
    2단계: 초기 클러스터들을 입력받아 seed/local 기준으로 최종 병합을 수행합니다.
    """
    if not initial_clusters: return []
    cluster_avg_planes = {i: fit_plane_pca(points[list(set(v for tri_idx in c for v in triangles[tri_idx]))])
                          for i, c in enumerate(initial_clusters)}
    tri_to_cluster_map = {tri_idx: i for i, c in enumerate(initial_clusters) for tri_idx in c}
    cluster_adjacencies = defaultdict(set)
    for i, cluster in enumerate(initial_clusters):
        for tri_idx in cluster:
            for neighbor_tri in adjacencies[tri_idx]:
                if neighbor_tri in tri_to_cluster_map:
                    neighbor_cluster_idx = tri_to_cluster_map[neighbor_tri]
                    if neighbor_cluster_idx != i:
                        cluster_adjacencies[i].add(neighbor_cluster_idx)
                        cluster_adjacencies[neighbor_cluster_idx].add(i)
    
    visited_clusters = [False] * len(initial_clusters)
    final_merged_clusters = []
    for i in range(len(initial_clusters)):
        if visited_clusters[i] or cluster_avg_planes.get(i) is None: continue
        
        seed_cluster_plane = cluster_avg_planes[i]
        stack, current_merged_group = [i], []
        visited_clusters[i] = True
        
        while stack:
            curr_cluster_idx = stack.pop()
            current_merged_group.extend(initial_clusters[curr_cluster_idx])
            curr_cluster_plane = cluster_avg_planes[curr_cluster_idx]

            for neighbor_cluster_idx in cluster_adjacencies[curr_cluster_idx]:
                if visited_clusters[neighbor_cluster_idx] or cluster_avg_planes.get(neighbor_cluster_idx) is None: continue
                
                neighbor_plane = cluster_avg_planes[neighbor_cluster_idx]
                cond_seed = are_coplanar(seed_cluster_plane, neighbor_plane, angle_deg_tolerance=merge_seed_angle_deg)
                cond_local = are_coplanar(curr_cluster_plane, neighbor_plane, angle_deg_tolerance=merge_local_angle_deg)
                
                if cond_seed and cond_local:
                    visited_clusters[neighbor_cluster_idx] = True
                    stack.append(neighbor_cluster_idx)
        final_merged_clusters.append(current_merged_group)
        
    return final_merged_clusters

# --------------------------------------------------------------------------
# 시각화 함수 (수정 없음)
# --------------------------------------------------------------------------
def visualize_without_light(mesh):
    """조명을 끄고 순수한 정점 색상만 확인하는 함수"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    
    # 렌더링 옵션을 가져와서 조명 효과(light_on)를 False로 설정
    opt = vis.get_render_option()
    opt.light_on = False
    
    print("조명을 끄고 시각화합니다. 양면이 모두 같은 색으로 보이는지 확인해 보세요.")
    vis.run()
    vis.destroy_window()
    
def visualize_with_open3d(points, triangles, clusters):
    tri_to_color = {}
    for cluster in clusters:
        color = np.random.rand(3)
        for tri_idx in cluster:
            tri_to_color[tri_idx] = color
            
    new_points = []
    new_triangles = []
    new_vertex_colors = []
    vertex_counter = 0
    
    for i, tri_indices in enumerate(triangles):
        color = tri_to_color.get(i, [0.5, 0.5, 0.5])
        
        p1 = points[tri_indices[0]]
        p2 = points[tri_indices[1]]
        p3 = points[tri_indices[2]]
        
        new_points.extend([p1, p2, p3])
        new_triangles.append([vertex_counter, vertex_counter + 1, vertex_counter + 2])
        vertex_counter += 3
        new_vertex_colors.extend([color, color, color])
        
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(new_points)),
        o3d.utility.Vector3iVector(np.array(new_triangles))
    )
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(new_vertex_colors))
    mesh.compute_vertex_normals()
    
    print("Open3D 윈도우를 엽니다. 닫으면 프로그램이 종료됩니다.")
    visualize_without_light(mesh) 


# --------------------------------------------------------------------------
# 메인 실행 함수 (새로운 워크플로우에 맞게 수정됨)
# --------------------------------------------------------------------------
def main():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])
    if not file_path: print("파일이 선택되지 않았습니다."); return
    df = pd.read_csv(file_path, header=None)
    
    points = []
    for _, row in df.iterrows():
        dist, az, el = row.iloc[0], np.radians(row.iloc[1]), np.radians(row.iloc[2])
        points.append([dist*np.cos(el)*np.cos(az), dist*np.cos(el)*np.sin(az), dist*np.sin(el)])
    points = np.array(points)

    # 1단계: 전체 삼각형과 '출신 정보' 생성
    print("전체 삼각형 및 출신 정보 생성 중...")
    triangles, triangle_origins = build_triangles_with_origin(df)
    print(f"총 {len(points)}개의 점과 {len(triangles)}개의 삼각형이 생성되었습니다.")
    
    # 2단계: 전체 메쉬에 대한 '완벽한 지도' 준비
    print("전체 메쉬 데이터(평면, 인접성) 준비 중...")
    adjacencies, triangle_planes = prepare_mesh_data(points, triangles)

    # 3단계: '같은 출신'끼리 1차 클러스터링 수행
    print("1단계: '같은 출신 지역' 내에서 초기 클러스터링 실행 중...")
    initial_clusters = cluster_by_origin(
        triangles, adjacencies, triangle_planes, triangle_origins,
        seed_angle_deg=20, local_angle_deg=20
    )
    print(f"초기 클러스터 개수: {len(initial_clusters)}")
    
    # 4단계: 1차 클러스터들을 대상으로 최종 병합 수행
    print("2단계: 초기 클러스터들을 대상으로 최종 병합 실행 중...")
    final_clusters = refine_and_merge_clusters(
        points, triangles, initial_clusters, adjacencies,
        merge_seed_angle_deg=20, merge_local_angle_deg=20
    )
    print(f"최종 클러스터 개수: {len(final_clusters)}")

    # 5단계: 최종 결과 시각화
    if final_clusters:
        print("시각화 준비 중...")
        visualize_with_open3d(points, triangles, final_clusters)
    else:
        print("시각화할 클러스터가 없습니다.")

if __name__ == "__main__":
    main()

