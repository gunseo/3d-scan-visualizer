import open3d as o3d
import numpy as np
import csv
import time

def cartesian_to_spherical(xyz_points):
    """
    (N, 3) 형태의 직교 좌표 배열을 구면 좌표 배열로 변환합니다.
    """
    x = xyz_points[:, 0]
    y = xyz_points[:, 1]
    z = xyz_points[:, 2]
    
    distance = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arcsin(z / (distance + 1e-9)))
    
    return np.stack([distance, azimuth, elevation], axis=1)

def generate_points_from_stl(filename, num_points, noise_level):
    """
    STL 파일에서 3D 메시를 읽어 표면에 점 구름을 샘플링하고 노이즈를 추가합니다.
    """
    try:
        # 1. STL 파일(메시) 불러오기
        mesh = o3d.io.read_triangle_mesh(filename)
        print(f"'{filename}' 파일을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: '{filename}' 파일을 읽는 중 문제가 발생했습니다: {e}")
        return None

    # 2. 메시 표면에서 균일하게 점 샘플링
    # 이것이 STL 방식의 가장 강력한 기능입니다!
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points_xyz = np.asarray(pcd.points)
    
    if points_xyz.size == 0:
        print("경고: STL 파일에서 점을 샘플링할 수 없습니다.")
        return None
        
    # 3. 노이즈 추가
    noise = (np.random.randn(*points_xyz.shape)) * noise_level
    noisy_points_xyz = points_xyz + noise
    
    # 4. 구면 좌표계로 변환
    points_spherical = cartesian_to_spherical(noisy_points_xyz)
    
    return points_spherical

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # --- 설정 값 ---
    # ## <<< 여기를 수정하여 사용하세요! >>> ##
    
    # 1. 입력할 STL 파일 이름
    STL_FILENAME = "case_1.stl"

    # 2. 생성할 총 점의 개수
    NUM_POINTS = 50000

    # 3. 노이즈 수준 (도면 단위 기준)
    NOISE_LEVEL = 0.05

    # 4. 저장할 CSV 파일 이름
    OUTPUT_FILENAME = "stl_scan_data.csv"
    
    # --- 데이터 생성 및 저장 ---
    start_time = time.time()
    
    spherical_data = generate_points_from_stl(STL_FILENAME, NUM_POINTS, NOISE_LEVEL)

    if spherical_data is not None:
        print(f"총 {len(spherical_data):,}개의 점 데이터 생성을 완료했습니다.")
        print("CSV 파일 저장을 시작합니다...")
        
        with open(OUTPUT_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['distance', 'azimuth', 'elevation'])
            writer.writerows([f'{val:.4f}' for val in row] for row in spherical_data)

        end_time = time.time()
        print(f"'{OUTPUT_FILENAME}' 파일이 성공적으로 생성되었습니다. (총 소요 시간: {end_time - start_time:.2f}초)")