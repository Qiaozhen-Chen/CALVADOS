import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

cwd = os.getcwd() #获取脚本运行时所在的文件夹绝对路径，赋值给变量 cwd；后续所有文件、输出目录都基于这个路径拼接。
sysname = 'polyR30' #定义模拟体系名称：polyR30，代表30 个精氨酸 RNA 多聚体（poly-RNA 30mer），文件夹、配置文件会带上这个标识区分不同模拟体系。

# set the side length of the cubic box
L = 30

# set the saving interval (number of integration steps)
N_save = 1000

# set final number of frames to save
N_frames = 1000

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293, # K
  ionic = 0.15, # molar，体系离子强度：0.15 mol/L（生理盐水浓度）。
  pH = 7.0,
  topol = 'center',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs，即运行时设置：轨迹写入间隔 wfreq，复用变量 N_save=1000；注释说明积分步单位 1 步 = 10 飞秒。
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
)

# PATH
path = f'{cwd}/{sysname}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')
#调用 Config 内置 write () 方法，将上面所有全局参数写入 yaml 配置文件；保存路径：./polyR30/config.yaml，这个文件是 CALVADOS 模拟器读取的核心全局参数文件。

components = Components(
  # Defaults
  molecule_type = 'rna',
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
  fresidues = f'{cwd}/residues_C2RNA.csv', # residue definitions，残基参数表文件路径：读取 csv 文件，里面存储 RNA 各碱基粗粒化粒子电荷、半径、相互作用参数。
  ffasta = f'{cwd}/rna.fasta',
  nmol = 1,
 
  # RNA settings
  rna_kb1 = 1400.0,
  rna_kb2 = 2200.0,
  #RNA 弯曲弹性力场参数 kb1、kb2：控制 RNA 链柔性，数值越大链越硬、越难弯折，CALVADOS 专属弯曲势常数。
  rna_ka = 4.20,
  rna_pa = 3.14,
  #RNA 角度扭转势参数 ka、pa：调控 RNA 螺旋扭转刚度。
  rna_nb_sigma = 0.4,
  #RNA 非键相互作用 LJ 势 σ 参数，单位 nm，控制粒子间范德华作用距离。
  rna_nb_scale = 15,
  #非键作用强度缩放系数，调控碱基间吸引 / 排斥强度，影响液液相分离行为。
  rna_nb_cutoff = 2.0
  #非键相互作用截断距离 2.0 nm，超过该距离不再计算粒子间范德华作用，节省算力。

)

components.add(name='polyR30')
components.write(path,name='components.yaml')

