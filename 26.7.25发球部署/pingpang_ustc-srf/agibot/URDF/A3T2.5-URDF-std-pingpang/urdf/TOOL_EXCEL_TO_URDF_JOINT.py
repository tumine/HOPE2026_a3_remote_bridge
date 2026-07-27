import os
import pandas as pd
import xml.etree.ElementTree as ET


def update_limits_and_axis(urdf_file: str,
                           excel_file: str,
                           output_file: str = 'URDF-JOINT.urdf'):
    """
    Excel 列：joint, effort, velocity, lower, upper, axis_x, axis_y, axis_z
    覆盖 URDF 中对应 <limit> 及 <axis xyz="...">
    """
    # ---- 读 Excel ----
    df = pd.read_excel(excel_file)
    df = df.fillna('')  # 把 NaN 变空字符串，避免 None

    # 建立字典：{joint_name: dict}
    data_map = {}
    for _, row in df.iterrows():
        j_name = str(row['joint']).strip()
        if not j_name:
            continue
        data_map[j_name] = {
            'effort':   str(row['effort'])   if str(row['effort'])   else None,
            'velocity': str(row['velocity']) if str(row['velocity']) else None,
            'lower':    str(row['lower'])    if str(row['lower'])    else None,
            'upper':    str(row['upper'])    if str(row['upper'])    else None,
            'axis_xyz': f"{row['axis_x']} {row['axis_y']} {row['axis_z']}".strip()
        }

    # ---- 解析 URDF ----
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    updated_cnt = 0
    for joint in root.findall('joint'):
        j_name = joint.get('name')
        if j_name not in data_map:
            continue

        cfg = data_map[j_name]

        # 1) 更新 <limit>
        limit_elem = joint.find('limit')
        if limit_elem is None:
            limit_elem = ET.SubElement(joint, 'limit')
        for key in ('effort', 'velocity', 'lower', 'upper'):
            if cfg[key] is not None:
                limit_elem.set(key, cfg[key])

        # 2) 更新 <axis xyz="...">
        axis_elem = joint.find('axis')
        if axis_elem is None:
            axis_elem = ET.SubElement(joint, 'axis')
        axis_elem.set('xyz', cfg['axis_xyz'])

        updated_cnt += 1

    # ---- 保存 ----
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"已更新 {updated_cnt} 个关节 → {os.path.abspath(output_file)}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    # 改为自己的绝对或相对路径
    update_limits_and_axis(
        urdf_file="0000014503_A3T2.5-URDF-std-pingpang-0409.urdf",
        excel_file="input_urdf_joint_2026-03-07.xlsx"
    )