import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# --- 데이터 처리 함수 ---
def spherical_to_cartesian_numpy(points):
    """구면 좌표계(거리, 방위각, 고도)를 직교 좌표계(x, y, z)로 변환합니다."""
    distance = points[:, 0]
    azimuth_deg = points[:, 1]
    elevation_deg = points[:, 2] + 90
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    return np.stack([x, y, z], axis=1)

# --- 엣지 검출 알고리즘 ---
def detect_edges(cloud, res=7.0, nearest_n=100):
    """
    포인트 클라우드에서 엣지를 검출하고, 엣지에 해당하는 점들의 '인덱스'를 반환합니다.
    """
    if cloud is None or len(cloud) == 0:
        return None

    print(f"엣지 분류 중... (res(λ)={res}, nearest_n={nearest_n})")
    if len(cloud) < nearest_n:
        nearest_n = len(cloud) - 1
        if nearest_n < 2: return None

    nbrs = NearestNeighbors(n_neighbors=nearest_n).fit(cloud)
    distances, indices = nbrs.kneighbors(cloud)

    z_i = distances[:, 1]
    
    v_i = cloud[indices]
    
    c_i = np.mean(v_i, axis=1)
    
    centroid_shift = np.linalg.norm(c_i - cloud, axis=1)
    
    edge_condition = centroid_shift > (res * (z_i + 1e-9))
    
    edge_indices = np.where(edge_condition)[0]
    
    if len(edge_indices) == 0:
        return None
        
    print(f"엣지 분류 완료. {len(edge_indices)}개의 엣지 포인트를 찾았습니다.")

    # [수정] 엣지 포인트의 좌표 대신 인덱스를 반환
    return edge_indices

# --- GUI 애플리케이션 클래스 ---
class ScanVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("3D 엣지 검출기")
        
        self.points_xyz = None

        main_frame = ttk.Frame(master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        file_frame = ttk.LabelFrame(left_panel, text="1. 데이터 불러오기", padding="10")
        file_frame.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(file_frame, text="파일 열기 (CSV/Excel/PCD)", command=self.load_from_file)
        self.load_button.pack(fill="x")
        
        param_frame = ttk.LabelFrame(left_panel, text="2. 파라미터 조정", padding="10")
        param_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(param_frame, text="전처리 다운샘플링 (↑수록 단순화)").grid(row=0, column=0, sticky="w", pady=2)
        self.downsample_var = tk.DoubleVar(value=0.01)
        ttk.Entry(param_frame, textvariable=self.downsample_var, width=10).grid(row=0, column=1)
        
        ttk.Separator(param_frame, orient='horizontal').grid(row=1, columnspan=2, sticky='ew', pady=5)

        ttk.Label(param_frame, text="엣지 민감도(λ) (↑수록 덜 검출)").grid(row=2, column=0, sticky="w", pady=2)
        self.res_var = tk.DoubleVar(value=7.0)
        ttk.Entry(param_frame, textvariable=self.res_var, width=10).grid(row=2, column=1)

        ttk.Label(param_frame, text="엣지 이웃 수").grid(row=3, column=0, sticky="w", pady=2)
        self.nearest_n_var = tk.IntVar(value=100)
        ttk.Entry(param_frame, textvariable=self.nearest_n_var, width=10).grid(row=3, column=1)
        
        action_frame = ttk.LabelFrame(left_panel, text="3. 실행", padding="10")
        action_frame.pack(fill="x")
        
        self.verify_button = ttk.Button(action_frame, text="전처리 확인", command=self.verify_preprocessing)
        self.verify_button.pack(fill="x", pady=(0,5))
        
        self.run_button = ttk.Button(action_frame, text="엣지 검출 실행", command=self.run_edge_detection)
        self.run_button.pack(fill="x")
        
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, sticky="ns")

        info_frame = ttk.LabelFrame(right_panel, text="정보", padding="10")
        info_frame.pack(fill="both", expand=True)
        self.point_count_label = ttk.Label(info_frame, text="불러온 점 개수: 0개")
        self.point_count_label.pack(anchor="w")
        self.processed_point_label = ttk.Label(info_frame, text="처리된 점 개수: 0개")
        self.processed_point_label.pack(anchor="w")

        self.clear_button = ttk.Button(right_panel, text="모든 데이터 초기화", command=self.clear_points)
        self.clear_button.pack(fill="x", pady=10, side="bottom")

    def load_from_file(self):
        file_path = filedialog.askopenfilename(
            title="데이터 파일 선택",
            filetypes=(("All Supported Files", "*.csv *.xlsx *.xls *.pcd"),("Point Cloud Data", "*.pcd"),("CSV 파일", "*.csv"),("Excel 파일", "*.xlsx *.xls"),("모든 파일", "*.*"))
        )
        if not file_path: return
        try:
            points = None
            if file_path.lower().endswith('.pcd'):
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
            elif file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                spherical_points = df.iloc[:, [0, 1, 2]].to_numpy()
                points = spherical_to_cartesian_numpy(spherical_points)
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
                spherical_points = df.iloc[:, [0, 1, 2]].to_numpy()
                points = spherical_to_cartesian_numpy(spherical_points)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다.")
            self.clear_points()
            self.points_xyz = points
            num_points = len(self.points_xyz)
            self.point_count_label.config(text=f"불러온 점 개수: {num_points}개")
            messagebox.showinfo("불러오기 성공", f"{num_points}개의 점을 성공적으로 불러왔습니다.")
        except Exception as e:
            messagebox.showerror("오류 발생", f"파일을 읽는 중 오류가 발생했습니다:\n{e}")

    def clear_points(self):
        self.points_xyz = None
        self.point_count_label.config(text="불러온 점 개수: 0개")
        self.processed_point_label.config(text="처리된 점 개수: 0개")
        print("모든 데이터가 초기화되었습니다.")

    def _preprocess_cloud(self):
        """내부용 전처리 함수: 다운샘플링을 수행"""
        if self.points_xyz is None: return None
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_xyz)

        voxel_size = self.downsample_var.get()
        if voxel_size > 0:
            print(f"다운샘플링 실행... Voxel Size: {voxel_size}")
            pcd_downsampled = pcd.voxel_down_sample(voxel_size)
            print(f"다운샘플링 후 포인트 수: {len(pcd_downsampled.points)}")
            return pcd_downsampled
        else:
            return pcd

    def verify_preprocessing(self):
        if self.points_xyz is None: return
        try:
            pcd_processed = self._preprocess_cloud()
            if pcd_processed is None: return

            num_original = len(self.points_xyz)
            num_processed = len(pcd_processed.points)
            self.processed_point_label.config(text=f"처리된 점 개수: {num_processed}개")

            messagebox.showinfo("전처리 확인", f"원본: {num_original}개\n전처리 후: {num_processed}개")

            pcd_processed.paint_uniform_color([0.2, 0.4, 1.0]) # 파란색
            o3d.visualization.draw_geometries([pcd_processed], window_name="전처리 결과 확인")

        except Exception as e:
            messagebox.showerror("오류 발생", f"전처리 확인 중 오류가 발생했습니다:\n{e}")

    def run_edge_detection(self):
        if self.points_xyz is None: return
        try:
            # 1. 전처리 실행
            pcd_processed = self._preprocess_cloud()
            if pcd_processed is None: return
            
            cloud_to_process = np.asarray(pcd_processed.points)
            self.processed_point_label.config(text=f"처리된 점 개수: {len(cloud_to_process)}개")

            # 2. 엣지 검출 실행
            res = self.res_var.get()
            nearest_n = self.nearest_n_var.get()
            edge_indices = detect_edges(cloud_to_process, res, nearest_n)
            
            if edge_indices is None or edge_indices.size == 0:
                messagebox.showwarning("결과 없음", "엣지를 찾지 못했습니다. 파라미터 값을 조정해 보세요."); return
            
            num_edges = len(edge_indices)
            messagebox.showinfo("검출 성공", f"엣지 {num_edges}개를 찾았습니다.")

            # 3. 시각화 (하나의 포인트 클라우드에 색상 지정)
            # 먼저 모든 점을 회색으로 칠할 색상 배열 생성
            colors = np.full_like(cloud_to_process, [0.8, 0.8, 0.8])
            
            # 엣지 인덱스에 해당하는 점들의 색상만 초록색으로 변경
            colors[edge_indices] = [0, 1, 0]
            
            # 포인트 클라우드에 색상 정보 할당
            pcd_processed.colors = o3d.utility.Vector3dVector(colors)

            # 단일 포인트 클라우드 객체 시각화
            o3d.visualization.draw_geometries([pcd_processed], window_name="엣지 검출 결과")
        
        except Exception as e:
            messagebox.showerror("오류 발생", f"검출 중 오류가 발생했습니다:\n{e}")

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ScanVisualizerApp(root)
    root.mainloop()

