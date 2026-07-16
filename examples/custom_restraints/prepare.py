import os
import pandas as pd
from calvados.cfg import Config, Job, Components 
#作用：从 CALVADOS 模拟框架的配置子模块导入 3 个核心类，是整套模拟的核心对象：
#Config：模拟全局参数类（盒子、温度、步长、保存频率、硬件、约束等），生成 config.yaml
#Job：集群调度任务类，用于生成提交脚本、提交 Slurm/PBS 任务（代码末尾注释使用）
#Components：体系组分类（蛋白分子数量、结构域约束、PDB 输入、力常数），生成 components.yaml
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
# --name：终端传参标识，运行时必须写 --name xxx
# nargs='?'：参数可传可不传，但配合 required=True 强制必须传入，nargs=1：必须传且只能传 1 个值，返回列表 ["p53"]（还要额外取元素，麻烦）；nargs='?'：最多 1 个值，返回单个字符串，适配文件名 / 蛋白名场景；nargs='+'：必须传至少 1 个值（适合一次性传入多个蛋白）
# required=True：必填参数，不输入程序直接报错退出
# type=str：传入内容强制转为字符串（蛋白名称）
args = parser.parse_args()
# 读取终端输入的所有参数，存入 args 对象；后续通过 args.name 获取用户输入的蛋白名。

cwd = os.getcwd()
sysname = f'{args.name:s}'

# set the side length of the cubic box
L = 40

# set the saving interval (number of integration steps)
N_save = 8000
#轨迹保存间隔：每 8000 个积分步输出一帧 DCD 轨迹文件，CALVADOS 默认单步 = 10 fs，所以每帧间隔 = 8000 × 10 fs = 80 ns。

# set final number of frames to save
N_frames = 4000
#总共要保存 4000 帧轨迹，用于后续分析。

residues_file = f'{cwd}/input/residues_CALVADOS3.csv'
#CALVADOS3 力场残基数据库，记录每种氨基酸粗粒化珠子数量、电荷、相互作用参数、疏水参数。

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293, # K
  ionic = 0.19, # molar
  pH = 7.0,
  topol = 'center',
  # 拓扑结构初始化模式：center 代表将输入 PDB 蛋白放置在模拟盒子中心。

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  threads = 4, #CPU 并行线程数，分配 4 个 CPU 核心运算。
  restart = 'checkpoint', #重启模式：checkpoint 代表模拟中断后可从 chk 断点文件续跑。
  frestart = 'restart.chk', #断点文件名称，续跑时读取该文件恢复体系坐标、速度。
  verbose = True,

  custom_restraints = True, #开启自定义谐波约束（结构域之间固定距离约束，用于维持蛋白三维折叠）。
  custom_restraint_type = 'harmonic', #约束势能类型：谐波势（简谐弹簧势，最常用）。
  fcustom_restraints = f'{cwd}/input/cres.txt', #自定义约束文件路径，文件内写结构域配对、目标距离、力常数。
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""

from calvados.analysis import save_conf_prop

save_conf_prop(path="{path:s}",name="{sysname:s}",residues_file="{residues_file:s}",output_path=f"{cwd}/data",start=100,is_idr=False,select='all')
"""
#start=100：跳过前 100 帧 equilibration 平衡阶段，从第 100 帧开始统计
#体系不是固有无序蛋白专属分析模式
#select='all'：体系内所有蛋白全部统计分析

config.write(path,name='config.yaml',analyses=analyses)
#调用 Config 类的 write 方法，在体系文件夹 path 下生成 config.yaml：包含所有上面设置的盒子、温度、步数、硬件参数；同时把上面定义的 analyses 后处理代码嵌入配置，模拟跑完自动执行分析。

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules，即体系内蛋白分子数量：1 个单体；做凝聚体多分子模拟可修改为 10、50 等。
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both，即多肽两端带电：N 端氨基、C 端羧基全部带电，也可选 N/C/both。
  # INPUT
  fresidues = residues_file, # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)，结构域定义文件路径：划分蛋白的不同功能结构域，设置域间谐波约束。
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files
  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go，两种分子内约束势能算法
  use_com = True, # apply on centers of mass instead of CA #约束作用对象：使用结构域质心计算距离，而非单颗 Cα 珠子，整体约束更稳定。
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2) ，即PAE 文件来源标识：1 = ColabFold 预测结构；0 代表 EBI 数据库原生 AlphaFold。
  k_harmonic = 700., # Restraint force constant #谐波约束力常数 700 kJ/(mol・nm²)，数值越大，结构域越难偏离天然距离。
)
components.add(name=args.name)

components.write(path,name='components.yaml')

# job = Job(envname='calvados-public')
# job.write(path,config,components)
# job.submit(path,njobs=1)
