import pandas as pd
from lxml import etree

# 读取 xlsx 文件
xlsx_path = r'C:\Users\admin\Siemens\swcache\URDF\A3-T2.5\0000014503_A3T2.5-URDF-std-pingpang-0519\urdf\input_urdf_link_2026-03-30.xlsx'
df = pd.read_excel(xlsx_path)
df = df.set_index('符号')

# 读取 urdf 文件，保留注释和格式
urdf_path = r'C:\Users\admin\Siemens\swcache\URDF\A3-T2.5\0000014503_A3T2.5-URDF-std-pingpang-0519\urdf\URDF-JOINT.urdf'
parser = etree.XMLParser(remove_blank_text=False)  # 保留空白文本
tree = etree.parse(urdf_path, parser)
root = tree.getroot()

for link in root.findall('link'):
    name = link.attrib.get('name')
    if name not in df.index:
        continue

    inertial = link.find('inertial')
    if inertial is None:
        continue

    # 质量
    mass = inertial.find('mass')
    if mass is not None and 'm' in df.columns:
        old_mass = mass.attrib.get('value')
        new_mass = str(df.loc[name, 'm'])
        if old_mass != new_mass:
            mass.attrib['value'] = new_mass

    # 重心
    origin = inertial.find('origin')
    if origin is not None and all(col in df.columns for col in ['X', 'Y', 'Z']):
        old_xyz = origin.attrib.get('xyz')
        new_xyz = f"{df.loc[name, 'X']} {df.loc[name, 'Y']} {df.loc[name, 'Z']}"
        if old_xyz != new_xyz:
            origin.attrib['xyz'] = new_xyz

    
	# 惯量
    inertia = inertial.find('inertia')
    if inertia is not None:
        for key in ['Lxx', 'Lyy', 'Lzz', 'Lxy', 'Lxz', 'Lyz']:
            if key in df.columns:
                inertia_key = key.lower().replace('l', 'i')
                old_val = inertia.attrib.get(inertia_key)
                new_val = str(df.loc[name, key])
                if old_val != new_val:
                    inertia.attrib[inertia_key] = new_val
    
	
	# ===== 新增：写入 rgba =====
    rgba_cols = ['R', 'G', 'B', 'A']
    if all(c in df.columns for c in rgba_cols):
        rgba_str = f"{df.loc[name, 'R']} {df.loc[name, 'G']} {df.loc[name, 'B']} {df.loc[name, 'A']}"

        # 找或建 <visual><material><color>
        visual = link.find('visual')
        if visual is None:
            visual = etree.SubElement(link, 'visual')

        material = visual.find('material')
        if material is None:
            material = etree.SubElement(visual, 'material')

        color = material.find('color')
        if color is None:
            color = etree.SubElement(material, 'color')

        color.attrib['rgba'] = rgba_str 
# 保存更正后的 urdf 文件，保留格式和注释
output_path = r'C:\Users\admin\Siemens\swcache\URDF\A3-T2.5\0000014503_A3T2.5-URDF-std-pingpang-0519\urdf\URDF-JOINT-LINK.urdf'
tree.write(output_path, encoding='utf-8', pretty_print=True, xml_declaration=True)
print(f'校对完成，已生成 {output_path}')