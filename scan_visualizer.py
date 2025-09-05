import tkinter as tk
from tkinter import filedialog, messagebox # filedialog 추가
import numpy as np
import pandas as pd # pandas 추가
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d # Open3D 라이브러리 불러오기

# --- 데이터 처리 함수들 ---

# 이전에 만들었던 대용량 데이터 처리용 NumPy 함수를 여기에 추가합니다.
def spherical_to_cartesian_numpy(points):
    """
    구면 좌표계 점들의 배열을 직교 좌표계로 한 번에 변환합니다.
    
    :param points: (N, 3) 형태의 NumPy 배열. 각 행은 (거리, 방위각, 고도각).
    :return: (N, 3) 형태의 (x, y, z) 좌표 NumPy 배열.
    """
    distance = points[:, 0]
    azimuth_deg = points[:, 1]
    elevation_deg = points[:, 2]
    
    azimuth_rad = np.radians(azimuth_deg)
    elevation_rad = np.radians(elevation_deg)
    
    r_on_xy_plane = distance * np.cos(elevation_rad)
    x = r_on_xy_plane * np.cos(azimuth_rad)
    y = r_on_xy_plane * np.sin(azimuth_rad)
    z = distance * np.sin(elevation_rad)
    
    return np.stack([x, y, z], axis=1)

# --- GUI 애플리케이션 클래스 정의 ---
class ScanVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("3D 스캔 데이터 시각화")
        master.geometry("600x250") # 윈도우 크기 조정

        self.current_points_xyz = np.array([]) 

        # --- 위젯(GUI 요소) 생성 ---
        # 파일 불러오기 프레임
        file_frame = tk.LabelFrame(master, text="데이터 불러오기", padx=10, pady=10)
        file_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # [변경] '파일에서 불러오기' 버튼을 새로 추가
        self.load_file_button = tk.Button(file_frame, text="파일에서 점 불러오기 (CSV 또는 Excel)", command=self.load_from_file)
        self.load_file_button.pack(fill="x") # pack을 사용해 프레임 내부에 배치

        # 시각화 및 초기화 프레임
        action_frame = tk.LabelFrame(master, text="작업", padx=10, pady=10)
        action_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ewns")

        self.plot_button = tk.Button(action_frame, text="전체 3D 플롯 보기", command=self.plot_all_points)
        self.plot_button.pack(fill="x", pady=5)
        
        self.clear_button = tk.Button(action_frame, text="모든 점 지우기", command=self.clear_points)
        self.clear_button.pack(fill="x", pady=5)
        
        # 정보 표시 프레임
        info_frame = tk.LabelFrame(master, text="정보", padx=10, pady=10)
        info_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ewns")

        self.point_count_label = tk.Label(info_frame, text="불러온 점 개수: 0개")
        self.point_count_label.pack(anchor="w")

        # Grid 레이아웃의 열(column) 확장 비율 설정
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
    
    # [신규] 파일 불러오기 기능을 담당하는 함수
    def load_from_file(self):
        """
        '파일에서 불러오기' 버튼을 눌렀을 때 실행되는 함수
        """
        # 1. 파일 선택 창 띄우기
        file_path = filedialog.askopenfilename(
            title="데이터 파일 선택",
            filetypes=( ("CSV 파일", "*.csv"),("Excel 파일", "*.xlsx *.xls"), ("모든 파일", "*.*"))
        )
        if not file_path: # 사용자가 '취소'를 누르면 아무것도 하지 않음
            return
        
        try:
            # 2. 파일 확장자에 따라 다르게 읽어오기
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # 3. 데이터가 비어있는지, 필요한 컬럼이 있는지 확인
            if df.empty or len(df.columns) < 3:
                raise ValueError("파일에 데이터가 없거나, 필요한 3개의 컬럼(거리, 수평각, 수직각)이 없습니다.")

            # 4. Pandas DataFrame을 NumPy 배열로 변환
            # 첫 세 개의 컬럼을 각각 거리, 수평각, 수직각 데이터로 가정
            spherical_points = df.iloc[:, [0, 1, 2]].to_numpy()

            # 5. NumPy 배열을 사용해 모든 점을 한 번에 XYZ로 변환
            self.current_points_xyz = spherical_to_cartesian_numpy(spherical_points)

            # 6. 사용자에게 결과 알려주기
            num_points = len(self.current_points_xyz)
            self.point_count_label.config(text=f"불러온 점 개수: {num_points}개")
            messagebox.showinfo("불러오기 성공", f"{num_points}개의 점을 성공적으로 불러왔습니다.\n'전체 3D 플롯 보기' 버튼을 눌러 확인하세요.")

        except Exception as e:
            messagebox.showerror("오류 발생", f"파일을 읽는 중 오류가 발생했습니다:\n{e}")

    def clear_points(self):
        self.current_points_xyz = np.array([])
        self.point_count_label.config(text="불러온 점 개수: 0개")
        messagebox.showinfo("초기화 완료", "모든 점이 지워졌습니다.")

    def plot_all_points(self):
        """
        저장된 모든 점들을 Open3D로 매우 빠르고 부드럽게 3D 시각화합니다.
        """
        if self.current_points_xyz.size == 0:
            # 이 부분은 GUI 프로그램에서는 messagebox를 쓰는 게 더 좋습니다.
            # print("시각화할 점이 없습니다.") 
            messagebox.showinfo("알림", "시각화할 점이 없습니다. 먼저 파일을 불러와주세요.")
            return

        # 1. NumPy 배열을 Open3D의 PointCloud 객체로 변환합니다.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.current_points_xyz)

        # 2. 색상을 추가하면 더 예쁘게 보입니다 (선택 사항)
        pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 모든 점을 회색으로

        # 3. 시각화 창을 띄웁니다.
        print("Open3D 창을 띄웁니다. 마우스로 회전/확대/축소가 가능합니다.")
        o3d.visualization.draw_geometries([pcd])

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ScanVisualizerApp(root)
    root.mainloop()