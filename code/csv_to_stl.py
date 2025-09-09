# -*- coding: utf-8 -*-
"""
3D 스캔 데이터(CSV) 클러스터링 최종 스크립트 (엄격한 라운드 기반 버전)

[작동 로직: 엄격한 라운드 기반 순차적 쌍 병합]
- 각 라운드(Pass)에서는 현재 라운드의 클러스터들만 대상으로 짝을 찾습니다.
- 한 클러스터는 한 라운드에서 최대 하나의 다른 클러스터와만 병합될 수 있습니다.
- 병합되어 생성된 새로운 클러스터는 현재 라운드에 참여하지 않고, 다음 라운드의 새로운 '선수'가 됩니다.
- 이 과정을 더 이상 새로운 쌍이 만들어지지 않을 때까지 반복합니다.
"""

import pandas as pd
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import open3d as o3d
from collections import defaultdict
from tqdm import tqdm

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

def load_points_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    points = []
    for _, row in df.iterrows():
        dist, az, el = row.iloc[0], np.radians(row.iloc[1]), np.radians(row.iloc[2])
        points.append([dist * np.cos(el) * np.cos(az), dist * np.cos(el) * np.sin(az), dist * np.sin(el)])
    return np.array(points), df

def build_triangles_with_origin(df):
    grid = {(int(row.iloc[1]), int(row.iloc[2])): idx for idx, row in df.iterrows()}
    triangles, triangle_origins = [], []
    az_steps = sorted(list(set(df.iloc[:, 1].astype(int))))
    el_steps = sorted(list(set(df.iloc[:, 2].astype(int))))
    for i, az in enumerate(az_steps):
        az_next = az_steps[(i + 1) % len(az_steps)]
        for j, el in enumerate(el_steps):
            el_next = el_steps[(j + 1) % len(el_steps)]
            p1_idx, p2_idx = grid.get((az, el)), grid.get((az_next, el))
            p3_idx, p4_idx = grid.get((az, el_next)), grid.get((az_next, el_next))
            if p1_idx is not None and p2_idx is not None and p3_idx is not None:
                triangles.append((p1_idx, p2_idx, p3_idx)); triangle_origins.append(i)
            if p3_idx is not None and p2_idx is not None and p4_idx is not None:
                triangles.append((p3_idx, p2_idx, p4_idx)); triangle_origins.append(i)
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
# 섹션 3: 핵심 클러스터링 알고리즘 (최종 로직)
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
        
        # ✨ 수정된 부분: show_progress가 True일 때만 Pass 시작 메시지 출력
        if show_progress:
            print(f"\n--- Pass {pass_num} 시작 (현재 클러스터 개수: {num_clusters}) ---")
        
        cluster_ids = sorted(list(clusters.keys()))
        cluster_planes = {cid: fit_plane_pca(points[list(set(p for t in clusters[cid] for p in triangles[t]))]) for cid in cluster_ids}

        merged_in_pass = set()
        
        # ✨ 수정된 부분: show_progress 값에 따라 tqdm을 사용하거나 사용하지 않음
        iterator = tqdm(cluster_ids, desc=f"  - Pass {pass_num} 진행 중") if show_progress else cluster_ids

        for cid in iterator:
            if cid in merged_in_pass or cluster_planes[cid][0] is None:
                continue

            for neighbor_id in list(graph.get(cid, set())):
                if neighbor_id in merged_in_pass or cluster_planes.get(neighbor_id, (None,))[0] is None:
                    continue
                
                dot_product = abs(np.dot(cluster_planes[cid][0], cluster_planes[neighbor_id][0]))
                if dot_product >= cos_thresh:
                    # (엣지 축약 로직은 이전과 동일)
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
# 섹션 4: 시각화 함수 (변경 없음)
# --------------------------------------------------------------------------
def visualize_without_light(mesh):
    vis = o3d.visualization.Visualizer(); vis.create_window(); vis.add_geometry(mesh)
    opt = vis.get_render_option(); opt.light_on = False; vis.run(); vis.destroy_window()

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


# --------------------------------------------------------------------------
# 섹션 5: 메인 실행부 (최종 로직에 맞게 수정)
# --------------------------------------------------------------------------
def main():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="CSV 파일을 선택하세요", filetypes=[("CSV files", "*.csv")])
    if not file_path: print("파일이 선택되지 않았습니다."); return

    points, df = load_points_from_csv(file_path)
    triangles, triangle_origins = build_triangles_with_origin(df)
    
    # --- ✨ 최초의 전체 삼각형 인접성 그래프 생성 (단 한 번만 실행) ---
    print("최초 인접성 그래프를 생성합니다 (전체 과정 중 한 번만 실행됩니다)...")
    edge_to_triangles = defaultdict(list)
    for i, tri in enumerate(tqdm(triangles, desc=" - 그래프 생성")):
        edges = [tuple(sorted(e)) for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]]
        for edge in edges: edge_to_triangles[edge].append(i)
    
    # 딕셔너리 형태의 그래프 {트라이앵글_id: {이웃_id_1, 이웃_id_2, ...}}
    adjacency_graph = defaultdict(set)
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            t1, t2 = tris
            adjacency_graph[t1].add(t2)
            adjacency_graph[t2].add(t1)
    
    print(f"총 {len(points)}개의 점과 {len(triangles)}개의 삼각형, {len(adjacency_graph)}개의 노드를 가진 그래프 생성 완료.")

    # --- 1단계: '같은 출신' 내에서 그래프 기반 병합 ---
    print("\n[1단계] '같은 출신' 그룹 내에서 병합을 시작합니다...")
    origin_groups = defaultdict(list)
    for i, origin_id in enumerate(triangle_origins): origin_groups[origin_id].append(i)

    clusters_from_stage1 = []
    for origin_id, tri_indices in tqdm(origin_groups.items(), desc="1단계 (출신 그룹별) 진행 중"):
        if not tri_indices: continue
        
        group_clusters = {idx: [idx] for idx in tri_indices}
        group_graph = {idx: {n for n in adjacency_graph.get(idx, set()) if n in group_clusters} for idx in tri_indices}
        
        merged = professional_graph_merge(points, triangles, group_clusters, group_graph, 20, show_progress=False)
        clusters_from_stage1.extend(merged)
    
    print(f"\n1단계 결과, 총 {len(clusters_from_stage1)}개의 초기 클러스터가 생성되었습니다.")

    # --- 2단계: 1단계 결과물들을 대상으로 그래프 기반 병합 ---
    print("\n[2단계] 초기 클러스터들을 대상으로 최종 병합을 시작합니다...")
    
    # 2-1. 1단계 결과물들로 새로운 클러스터와 그래프 생성
    s2_clusters = {i: c for i, c in enumerate(clusters_from_stage1)}
    tri_to_cid_map = {tri_idx: cid for cid, cluster in s2_clusters.items() for tri_idx in cluster}
    
    print("  - 2단계용 인접 그래프를 생성합니다...")
    s2_graph = defaultdict(set)
    for cid, cluster in tqdm(s2_clusters.items(), desc="  - 2단계 그래프 생성 중"):
        for tri_idx in cluster:
            for neighbor_tri in adjacency_graph.get(tri_idx, set()):
                neighbor_cid = tri_to_cid_map.get(neighbor_tri)
                if neighbor_cid is not None and neighbor_cid != cid:
                    s2_graph[cid].add(neighbor_cid)
    
    # 2-2. 최종 병합 실행
    final_clusters = professional_graph_merge(points, triangles, s2_clusters, s2_graph, 45,show_progress=True)
    
    print(f"\n최종 클러스터 {len(final_clusters)}개가 생성되었습니다.")

    # --- 3단계: 시각화 ---
    if final_clusters:
        visualize_clusters(points, triangles, final_clusters)

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("'tqdm' 라이브러리가 필요합니다. 'pip install tqdm' 명령어로 설치해주세요.")
        exit()
    main()