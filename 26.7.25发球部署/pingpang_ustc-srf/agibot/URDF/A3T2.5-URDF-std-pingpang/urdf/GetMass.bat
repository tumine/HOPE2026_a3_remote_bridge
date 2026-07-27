@echo off
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  "$xml=[xml](gc (ls *.urdf).FullName);" ^
  "$xml.robot.link | ?{$_.inertial} | %% {" ^
  "  $o=$_.inertial.origin; $i=$_.inertial; $m=$i.mass.value;" ^
  "  $xyz=$o.xyz -split '\s+'; $rpy=$o.rpy -split '\s+';" ^
  "  'link={0} ox={1} oy={2} oz={3} rx={4} ry={5} rz={6} mass={7} ixx={8} ixy={9} ixz={10} iyy={11} iyz={12} izz={13}' -f" ^
  "  $_.name,$xyz[0],$xyz[1],$xyz[2],$rpy[0],$rpy[1],$rpy[2]," ^
  "  $m,$i.inertia.ixx,$i.inertia.ixy,$i.inertia.ixz,$i.inertia.iyy,$i.inertia.iyz,$i.inertia.izz" ^
  "} | Out-File -Encoding UTF8 mass.txt"
echo 已生成 mass.txt
pause