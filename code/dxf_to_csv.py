import ezdxf
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

def generate_edge_scan_from_dxf(filename, point_density, spread, noise_level):
    """
    3D DXF 파일의 모서리를 따라가며, 각 지점 주변에 5개의 점을 생성합니다.
    """
    try:
        doc = ezdxf.readfile(filename)
        msp = doc.modelspace()
        print(f"'{filename}' 파일을 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"오류: '{filename}' 파일을 읽는 중 문제가 발생했습니다: {e}")
        return None

    # 1. DXF에서 모든 3D 선분(segment) 정보 추출
    all_segments = []
    for entity in msp:
        try:
            points_to_add = []
            if entity.dxftype() == 'LINE':
                points_to_add = [entity.dxf.start, entity.dxf.end]
            elif entity.dxftype() in {'POLYLINE', '3DPOLYLINE', 'LWPOLYLINE'}:
                points_to_add = [p for p in entity.points()]
            elif entity.dxftype() in {'ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE'}:
                segment_length = 1.0 / point_density
                flattened_points = list(entity.flattening(distance=segment_length))
                points_to_add = [p for p in flattened_points]

            if len(points_to_add) > 1:
                for i in range(len(points_to_add) - 1):
                    all_segments.append(
                        (np.array(points_to_add[i]), np.array(points_to_add[i+1]))
                    )
        except (AttributeError, TypeError, ValueError):
            continue
    
    if not all_segments:
        print("경고: DXF 파일에서 점을 추출할 수 있는 객체를 찾지 못했습니다.")
        return None

    final_points_xyz = []

    # 2. 각 선분을 따라가며 중심점 및 주변 점 5개 생성
    for start_node, end_node in all_segments:
        segment_vector = end_node - start_node
        segment_length = np.linalg.norm(segment_vector)
        
        if segment_length < 1e-6: # 길이가 거의 없는 선분은 건너뛰기
            continue
            
        # 모서리 방향 벡터 (정규화)
        edge_direction = segment_vector / segment_length
        
        # 모서리 위에 생성할 중심점의 개수 결정
        num_center_points = max(2, int(segment_length * point_density))
        center_points = np.linspace(start_node, end_node, num_center_points)

        # 3. 각 중심점에 대해 수직 방향 벡터 2개 계산
        for center_point in center_points:
            # 수직 벡터를 찾기 위한 임의의 벡터 선택 (모서리 방향과 평행하지 않도록)
            if np.allclose(np.abs(edge_direction), [0, 0, 1]):
                ref_vector = np.array([1, 0, 0])
            else:
                ref_vector = np.array([0, 0, 1])

            # 외적(Cross Product)을 이용해 첫 번째 수직 벡터 계산
            side1_vector = np.cross(edge_direction, ref_vector)
            side1_vector /= np.linalg.norm(side1_vector) # 정규화

            # 두 번의 외적을 통해 두 번째 수직 벡터 계산
            side2_vector = np.cross(edge_direction, side1_vector)
            # 이미 정규화되어 있음

            # 4. 5개의 점 생성 (중심 1 + 양옆 4)
            # 중심점 추가
            final_points_xyz.append(center_point)
            
            # 양옆 4개 점 추가
            final_points_xyz.append(center_point + side1_vector * spread)
            final_points_xyz.append(center_point - side1_vector * spread)
            final_points_xyz.append(center_point + side2_vector * spread)
            final_points_xyz.append(center_point - side2_vector * spread)

    if not final_points_xyz:
        print("경고: 최종 점을 생성하지 못했습니다.")
        return None

    final_points_xyz = np.array(final_points_xyz)
    
    # 5. 최종 노이즈 추가
    noise = (np.random.rand(*final_points_xyz.shape) - 0.5) * noise_level
    noisy_points_xyz = final_points_xyz + noise
    
    # 6. 구면 좌표계로 변환
    points_spherical = cartesian_to_spherical(noisy_points_xyz)
    
    return points_spherical


# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # --- 설정 값 ---
    # ## <<< 여기를 수정하여 사용하세요! >>> ##
    
    # 1. 입력할 3D DXF 파일 이름
    DXF_FILENAME = "case_1 v1.dxf"

    # 2. 모서리를 따라 찍을 점의 밀도 (값이 클수록 빽빽해짐)
    POINT_DENSITY = 2.0

    # 3. 모서리 양옆으로 퍼지는 거리
    SPREAD = 1.0

    # 4. 최종 점들에 추가될 미세한 노이즈 수준
    NOISE_LEVEL = 0.3

    # 5. 저장할 CSV 파일 이름
    OUTPUT_FILENAME = "dxf_edge_scan_precise.csv"
    
    # --- 데이터 생성 및 저장 ---
    start_time = time.time()
    
    spherical_data = generate_edge_scan_from_dxf(DXF_FILENAME, POINT_DENSITY, SPREAD, NOISE_LEVEL)

    if spherical_data is not None:
        print(f"총 {len(spherical_data):,}개의 점 데이터 생성을 완료했습니다.")
        print("CSV 파일 저장을 시작합니다...")
        
        with open(OUTPUT_FILENAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['distance', 'azimuth', 'elevation'])
            writer.writerows([f'{val:.4f}' for val in row] for row in spherical_data)

        end_time = time.time()
        print(f"'{OUTPUT_FILENAME}' 파일이 성공적으로 생성되었습니다. (총 소요 시간: {end_time - start_time:.2f}초)")