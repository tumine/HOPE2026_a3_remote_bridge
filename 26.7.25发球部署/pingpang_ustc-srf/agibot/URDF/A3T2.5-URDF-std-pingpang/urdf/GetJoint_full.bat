@echo off
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  "$f=(Get-ChildItem *.urdf).FullName;" ^
  "$xml=[xml](Get-Content $f -Raw);" ^
  "'joint,effort,velocity,origin_x,origin_y,origin_z,r,p,y,axis_x,axis_y,axis_z,limit_lower,limit_upper';" ^
  "$xml.robot.joint | ForEach-Object {" ^
  "  $l=$_.limit; $o=$_.origin; $ax=$_.axis;" ^
  "  $ori=($o.xyz -split '\s+'); if(!$ori){$ori=@('0','0','0')};" ^
  "  $rpy=($o.rpy -split '\s+'); if(!$rpy){$rpy=@('0','0','0')};" ^
  "  $axis=($ax.xyz -split '\s+'); if(!$axis){$axis=@('0','0','0')};" ^
  "  $eff=if($l.effort){$l.effort}else{''};" ^
  "  $vel=if($l.velocity){$l.velocity}else{''};" ^
  "  $low=if($l.lower){$l.lower}else{''};" ^
  "  $upp=if($l.upper){$l.upper}else{''};" ^
  "  ('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}' -f" ^
  "   $_.name,$eff,$vel,$ori[0],$ori[1],$ori[2],$rpy[0],$rpy[1],$rpy[2],$axis[0],$axis[1],$axis[2],$low,$upp)" ^
  "} | Out-File -Encoding UTF8 joints.txt"
echo 已生成 joints.txt
pause