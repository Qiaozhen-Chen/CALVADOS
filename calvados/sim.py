#RNA会完整用到这里的所有力场 / 相互作用，还额外增加 RNA 专属的角度弯曲势（sim.py line390~)

import numpy as np
import openmm
from openmm import app, unit

from datetime import datetime

import mdtraj as md

from tqdm import tqdm
import os

from calvados import build, interactions
#导入 CALVADOS 自定义模块：build：盒子构建、分子网格排布、双层膜坐标生成、随机放置分子；interactions：所有粗粒化力场、 restraints、DH 静电、脂质专属作用、自定义约束工厂函数

from yaml import safe_load
#安全读取 yaml 配置文件，解析 config.yaml/components.yaml 字典参数。

from Bio.SeqUtils import seq3
#Biopython 工具：氨基酸单字母序列转三字母残基名（A→ALA、K→LYS），用于 PDB 拓扑残基命名。

from .components import *

#定义模拟主类 Sim，封装读配置 → 构建分子组分 → 搭建 OpenMM 体系 → 放置粒子 → 添加全部力场 → 运行模拟完整流程。
class Sim:
#init 构造函数（初始化参数）：path：模拟输出文件夹路径；config：config.yaml 解析后的全局参数字典；components：components.yaml 解析后的分子组分定义字典
    def __init__(self,path,config,components):
        """
        simulate openMM Calvados;
        parameters are provided by config dictionary """

        self.path = path
        # parse config
        for key, val in config.items():
            setattr(self, key, val)
#遍历配置字典所有键值，动态给 Sim 实例创建属性。例 yaml 写 temp: 300 → 代码可直接用 self.temp。

        for key, val in components['defaults'].items():
            setattr(self, f'default_{key}', val)

        self.comp_dict = components['system']
        self.comp_defaults = components['defaults']

        self.box = np.array(self.box)
        self.eps_lj *= 4.184 # kcal to kJ/mol #单位换算：输入 LJ 势阱深度单位 kcal/mol，乘以系数转为 OpenMM 统一使用的 kJ/mol。

        if self.restart == 'checkpoint' and os.path.isfile(f'{self.path}/{self.frestart}'):
            self.slab_eq = False
            self.bilayer_eq = False
        #重启逻辑：如果从 checkpoint 续跑且检查点文件存在，跳过 slab / 双层膜平衡步骤（已经平衡过了）。

        if self.slab_eq:
            self.rcent = interactions.init_slab_restraints(self.box,self.k_eq)
        #开启 slab 平衡模式：创建 z 方向限制蛋白 / RNA 在中间薄层的外部约束力，存入 self.rcent。

        if self.ext_force:
            self.rcent = openmm.CustomExternalForce(self.ext_force_expr)
        #自定义外部势场：读取 yaml 自定义势能表达式，生成通用自定义外力对象覆盖 slab 约束。

    def make_components(self):
        self.components = np.empty(0) #初始化空 numpy 数组，用来存放所有分子组分实例（Protein/Lipid/RNA...）。
        self.use_restraints = False   
###标记体系是否包含 GO 拓扑约束（蛋白折叠约束），默认 False。
###ps: GO 拓扑折叠约束确实是在 prepare.py 里配置开关，不是在 Sim 模拟主代码里硬写死，prepare.py 控制「是否开启约束、约束类型是 harmonic /go、力常数、PAE 参数」，最终写入 components.yaml；
###ps: Sim 类代码只是读取 yaml 并实例化约束力，不负责开关控制。
       
        # comp_setup = 'spiral' if self.topol=='shift_ref_bead' else 'linear'
        for name, properties in self.comp_dict.items(): #循环遍历 components.yaml system 下每一种分子（如 proteinA、POPC、mRNA1）。
            molecule_type = properties.get('molecule_type', self.default_molecule_type) #读取当前分子类型，若无则使用全局默认分子类型。
            if molecule_type == 'protein':
                # Protein component
                comp_setup = 'compact'
                comp = Protein(name, properties, self.comp_defaults) #蛋白：紧凑构象初始化，实例化 Protein 类对象。
            elif molecule_type in ['lipid','cooke_lipid']:
                # Lipid component
                comp_setup = 'linear'
                comp = Lipid(name, properties, self.comp_defaults)。 #标准脂质 / Cooke 粗粒化脂质：线性链排布，Lipid 对象。
            elif molecule_type in ['crowder']:
                # Crowder component
                comp_setup = 'compact'
                comp = Crowder(name, properties, self.comp_defaults)  #拥挤物（球形大分子 crowding agent），紧凑球形初始化。
            elif molecule_type in ['rna']:
                # Crowder component
                comp_setup = 'spiral'
                comp = RNA(name, properties, self.comp_defaults)      #RNA：螺旋构象初始化，RNA 对象。
            elif molecule_type == 'cyclic':
                comp_setup = 'compact'
                comp = Cyclic(name, properties, self.comp_defaults)    #环状多肽 / 环状核酸，紧凑环结构。
            elif molecule_type == 'seastar':
                comp_setup = 'compact'
                comp = Seastar(name, properties, self.comp_defaults)   #海星型多臂大分子（多结构域分支蛋白）。
            elif molecule_type == 'ptm_protein':
                comp_setup = 'compact'
                comp = PTMProtein(name, properties, self.comp_defaults) #带翻译后修饰（磷酸化、泛素化等）的蛋白子类。
            else:
                # Generic component
                comp_setup = 'linear'
                comp = Component(name, properties, self.comp_defaults)  #未定义类型默认通用线性链组分。

            comp.eps_lj = self.eps_lj    #全局 LJ 势阱深度赋值给当前分子对象。
            comp.calc_properties(pH=self.pH, verbose=self.verbose, comp_setup=comp_setup) #分子内部计算：根据 pH 计算每个珠子电荷、珠子半径 sigma、lambda 混合参数、初始坐标、成键拓扑、残基序列。
            if comp.restraint:
                if comp.restraint_type == 'go':
                    comp.init_restraint_force(
                        eps_lj=self.eps_lj, cutoff_lj=self.cutoff_lj,
                        eps_yu=self.eps_yu, k_yu = self.k_yu
                    )   #如果该分子开启 GO 拓扑约束（折叠约束）：GO 约束需要静电 / LJ 参数，传入初始化约束力。
                else:
                    comp.init_restraint_force()   #普通距离约束，无需额外参数。
                self.use_restraints = True        #全局标记体系存在约束。

            self.components = np.append(self.components, comp)    #将当前分子实例存入总组分数组。

    #count_components () 统计分子数量、重排组分顺序
    def count_components(self):
        """ Count components and molecules. """

        self.ncomponents = 0
        self.nmolecules = 0
    #初始化计数器：组分种类数、总分子数量。
        
        for comp in self.components:
            self.ncomponents += 1
            self.nmolecules += comp.nmol
    #遍历累加：每种分子有 nmol 个拷贝，全部求和。

        print(f'Total number of components in the system: {self.ncomponents}')
        print(f'Total number of molecules in the system: {self.nmolecules}')
    #控制台打印统计信息。

        # move lipids at the end of the array
        molecule_types = np.asarray([c.molecule_type for c in self.components])
        #提取所有组分类型生成数组，用于筛选分类。
        self.nlipids = np.sum([c.nmol if c.molecule_type == 'lipid' else 0 for c in self.components])
        self.ncookelipids = np.sum([c.nmol if c.molecule_type == 'cooke_lipid' else 0 for c in self.components])
        self.nproteins = np.sum([c.nmol if c.molecule_type == 'protein' else 0 for c in self.components])
        self.ncrowders = np.sum([c.nmol if c.molecule_type == 'crowder' else 0 for c in self.components])
        self.nrnas = np.sum([c.nmol if c.molecule_type == 'rna' else 0 for c in self.components])
        #分别统计各类分子总拷贝数，存为实例变量供后续判断脂质 / 蛋白体系分支。

        if ((self.ncomponents > 1) or (self.nmolecules > 1)) and (self.topol in ['single', 'center']):
            raise ValueError("Topol 'center' incompatible with multiple molecules.")
###如果在prepare.py设置 topol="center"：强制只能 1 种分子、1 个拷贝，多分子直接报错；
###如果在prepare.py设置 topol="grid" / topol="slab"：无数量限制，随便混合蛋白 / RNA / 脂质 / 拥挤物，几十上百个分子都没问题。
#slab：人为预构建两相界面，成核快。蛋白 / RNA 初始集中在 z 中层，局部浓度瞬间拉高，极易快速形成致密凝聚液滴，适合研究两相共存、界面性质、液滴融合、蛋白 RNA 共凝聚；适用场景：IDR 液液相分离、蛋白 - RNA 共相分离、细胞内凝聚体模拟。
#grid：全局低均匀浓度，成核慢。分子均匀稀释在整个盒子，局部浓度低，相互碰撞概率低；弱粘滞 IDR 可能几十万步都看不到明显液滴；适用场景：单分子折叠、稀溶液寡聚、无分相的本体均匀溶液，不适合 LLPS。

        # move proteins at the beginning of the array
        if self.nmolecules > self.nproteins:
            protein_components = self.components[np.where(molecule_types=='protein')]
            non_protein_components = self.components[np.where(molecule_types!='protein')]
            self.components = np.append(protein_components,non_protein_components)

    def build_system(self):    #整个模拟体系构建入口，四大任务：初始化拓扑 / 盒子、生成所有分子、初始化全部粗粒化力、放置粒子并添加键 / 角 / 约束。
        """
        Set up system
        * component definitions
        * build particle coordinates
        * define interactions
        * set restraints
        """

        self.top = md.Topology()   #新建 MDTraj 空拓扑对象，用来记录所有链、残基、珠子、键，最终输出 PDB。
        self.system = openmm.System()  #新建 OpenMM 核心 System 对象，存储所有粒子质量、所有作用力、周期性盒子。
        a, b, c = build.build_box(self.box[0],self.box[1],self.box[2])
        self.system.setDefaultPeriodicBoxVectors(a, b, c)  #根据盒子长宽高生成三维周期盒子矢量，赋值给 OpenMM 体系。


        # init interaction parameters (required before make components)
        self.eps_yu, self.k_yu = interactions.genParamsDH(self.temp,self.ionic)  #生成 Debye-Hückel 静电（Yu 势）参数：屏蔽强度、势阱深度，依赖温度与离子强度。

        # make components, 先生成所有分子对象，再统计数量、重排顺序。
        self.make_components()
        self.count_components()

        # init interactions, 初始化两大非键作用：ah Ashbaugh-Hatch LJ 混合势（粗粒化范德华）; yu Debye-Hückel 静电势
        self.ah, self.yu = interactions.init_nonbonded_interactions(
            self.eps_lj,self.cutoff_lj,self.eps_yu,self.k_yu,self.cutoff_yu,self.fixed_lambda
            )
        if self.nlipids > 0:
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=1.9
            )            #存在标准脂质时，初始化脂质专属作用力：cos 脂质尾部疏水聚集势; cn 电荷 - 非极性脂质交叉作用势; factor=1.9 标准脂质作用缩放系数。
        if self.ncookelipids > 0:
            if self.nlipids > 0:
                raise     #Cooke 脂质与标准脂质不能共存；作用缩放系数改为 3.0 适配 Cooke 粗粒模型。
###像蛋白质，rna等上面定义的其他类型的分子，关于缩放系数具体的值，在prepare.py文件里设置。如果体系有蛋白设置到idr区域，可以对蛋白进行局部约束和分区域设置缩放系数。
            
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=3.0
            )

        self.nparticles = 0 # bead counter
        self.grid_counter = 0 # molecule counter for xy and xyz grids

        self.pos = []

        if self.topol == 'slab': # proteins + rna
    #slab 薄层拓扑：蛋白 + RNA 排布在盒子 z 轴中间薄层，生成均匀三维网格坐标并平移到盒子中心 z 区间。
    #拥挤物分两部分排布：薄层上下两侧区域，避免和中间蛋白 RNA 重叠。
            self.xyzgrid = build.build_xyzgrid(self.nproteins+self.nrnas,[self.box[0],self.box[1],self.slab_width])
            self.xyzgrid += np.asarray([0,0,self.box[2]/2.-self.slab_width/2.])
            if self.ncrowders > 0: # crowder
                xyzgrid = build.build_xyzgrid(np.ceil(self.ncrowders/2.),[self.box[0],self.box[1],self.box[2]/2.-self.slab_outer])
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid, axis=0)
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.slab_outer]), axis=0)
        elif self.topol == 'grid':
        #grid 均匀网格拓扑：所有分子均匀铺满整个三维盒子网格。
        #存在脂质：生成 xy 平面网格用于双层膜排布，数量 ×1.05 预留冗余位置防止重叠。
        #膜体系蛋白 / RNA 排布在双层膜上下水相区域，分上下两堆网格。
        #Cooke 脂质双层膜排布逻辑同上。
            self.xyzgrid = build.build_xyzgrid(self.nmolecules,self.box)
        if self.nlipids > 0:
            self.bilayergrid = build.build_xygrid(int(self.nlipids*1.05),self.box)
            if (self.nproteins + self.nrnas) > 0:
                xyzgrid = build.build_xyzgrid(np.ceil((self.nproteins+self.nrnas)/2.),[self.box[0],self.box[1],self.box[2]/2.-self.box[0]])
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)
        if self.ncookelipids > 0:
            self.bilayergrid = build.build_xygrid(int(self.ncookelipids*1.05),self.box)
            if (self.nproteins + self.nrnas) > 0:
                xyzgrid = build.build_xyzgrid(np.ceil((self.nproteins+self.nrnas)/2.),[self.box[0],self.box[1],self.box[2]/2.-self.box[0]])
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)
       
        #循环所有分子，逐个放入体系
        for cidx, comp in enumerate(self.components):
            for idx in range(comp.nmol):
                if self.verbose:
                    print(f'Component {cidx}, Molecule {idx}: {comp.name}')
                # particle definitions
                #两步基础构建：1. add_mdtraj_topol：把该分子的残基、珠子、键写入 MDTraj 拓扑；2. add_particles_system：把每个珠子质量加入 OpenMM System
                self.add_mdtraj_topol(comp)
                self.add_particles_system(comp.mws)

                # add interactions + restraints
                #区分放置函数：蛋白 / RNA / 拥挤物：place_molecule 网格 / 随机放置；脂质：place_bilayer 嵌入双层膜平面
                if comp.molecule_type in ['protein','crowder','cyclic','seastar','ptm_protein']:
                    xs = self.place_molecule(comp)
                elif comp.molecule_type in ['lipid','cooke_lipid']:
                    xs = self.place_bilayer(comp)
                elif comp.molecule_type == 'rna':
                    xs = self.place_molecule(comp)
                    
                #给当前分子所有珠子添加 LJ、静电、脂质专属作用、成键、角、内部约束。 
                self.add_interactions(comp)

                # add restraints towards box center
                #开启薄层平衡 / 自定义外力，且分子允许外部约束时，给该分子所有珠子添加中心限制外力。
                if (self.slab_eq or self.ext_force) and comp.ext_restraint:
                    self.add_ext_restraints(comp)

        #yaml 配置自定义跨分子距离约束：先解析约束文件映射全局珠子索引，再添加约束力。
        if self.custom_restraints:
            self.map_custom_restraints()
            self.add_custom_restraints()

        self.pdb_cg = f'{self.path}/top.pdb'   #粗粒化初始拓扑 PDB 输出路径。
        a = md.Trajectory(self.pos, self.top, 0, self.box, [90,90,90]) #用全部粒子坐标 + 拓扑生成 MDTraj 轨迹对象，正交盒子 90° 夹角。
        if self.restart != 'pdb': # only save new topology if no system pdb is given
            a.save_pdb(self.pdb_cg)

        self.add_forces_to_system()
        self.print_system_summary()

#add_forces_to_system () 将所有力注册进 OpenMM 体系
    def add_forces_to_system(self):
        """ Add forces to system. """   #先加分子间非键：静电 yu、LJ 混合 ah。

        # Intermolecular forces
        for force in [self.yu, self.ah]:
            self.system.addForce(force)  #膜体系额外添加脂质聚集、电荷 - 脂质作用。

        if (self.nlipids > 0) or (self.ncookelipids > 0):
            for force in [self.cos, self.cn]:
                self.system.addForce(force)

        # Intramolecular forces
        for comp in self.components:
            comp.get_forces() # bonded, angles, restraints...
            for force in comp.forces:
                self.system.addForce(force)
            if comp.restraint:
                print(f'Number of restraints for comp {comp.name}: {comp.cs.getNumBonds()}')    #添加分子内作用：共价键、角、GO 折叠约束，并打印约束数量。

        # External force
        if self.ext_force:
            self.system.addForce(self.rcent)    #自定义全局外部势场。

        # Equilibration forces
        if self.slab_eq:
            self.system.addForce(self.rcent)    #薄层平衡 z 方向限制力。

        # Custom forces
        if self.custom_restraints:
            self.system.addForce(self.cres)
            print(f'Number of custom restraints: {self.cres.getNumBonds()}')   #跨分子自定义距离约束。

        # Barostat force
        if self.box_eq:
            barostat = openmm.openmm.MonteCarloAnisotropicBarostat(
                    [self.pressure[0]*unit.bar,self.pressure[1]*unit.bar,self.pressure[2]*unit.bar],
                    self.temp*unit.kelvin,self.boxscaling_xyz[0],self.boxscaling_xyz[1],
                    self.boxscaling_xyz[2],1000)
            self.system.addForce(barostat)        #各向异性蒙特卡洛控压杆，三方向独立控压，用于本体溶液体系 NPT 平衡。

        # Bilayer eq. force
        if self.bilayer_eq:
            barostat = openmm.openmm.MonteCarloMembraneBarostat(self.pressure[0]*unit.bar,
                    0*unit.bar*unit.nanometer, self.temp*unit.kelvin,
                    openmm.openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.openmm.MonteCarloMembraneBarostat.ZFixed, 10000)
            self.system.addForce(barostat)        #膜专属控压杆：xy 平面各向同性、z 轴固定，维持膜零横向张力。

   #print_system_summary () 体系信息输出 + 序列化 System
    def print_system_summary(self, write_xml: bool = True):
        """ Print system information and write xml. """

        if write_xml:
            with open(f'{self.path}/{self.sysname}.xml', 'w') as output:
                output.write(openmm.XmlSerializer.serialize(self.system))
     #将完整 OpenMM 体系（粒子、力、盒子）序列化为 xml 文件，方便重启、复现体系。
        
        print(f'{self.nparticles} particles in the system')
        print('---------- FORCES ----------')
        print(f'ah: {self.ah.getNumParticles()} particles, {self.ah.getNumExclusions()} exclusions')
        print(f'yu: {self.yu.getNumParticles()} particles, {self.yu.getNumExclusions()} exclusions')
        if self.slab_eq:
            print(f'Equilibration restraints (rcent) towards box center in z direction')
            print(f'rcent: {self.rcent.getNumParticles()} restraints')
        if self.bilayer_eq:
            print(f'Equilibration under zero lateral tension')
        if self.box_eq:
            print(f'Equilibration through changes in box side lengths along '+' and '.join(np.array(['X','Y','Z'])[self.boxscaling_xyz]))

    #place_molecule () 蛋白 / RNA / 拥挤物坐标排布
    def place_molecule(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """

        if self.topol == 'slab':     #slab 模式：取出网格基准坐标，叠加分子自身初始构象坐标，网格计数 + 1。
            x0 = self.xyzgrid[self.grid_counter]
            # x0[2] = self.box[2] / 2. # center in z
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'grid':   #grib 模式同slab 模式
            x0 = self.xyzgrid[self.grid_counter]
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'center':  #center 拓扑：唯一分子放盒子几何中心。
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
        elif self.topol == 'shift_ref_bead':  #以指定参考珠子对齐盒子中心，分子整体平移消除参考珠子偏移。
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
            xs -= comp.xinit[self.ref_bead]
        else:                                #其余拓扑：随机投放，最多尝试 10000 次避免粒子重叠。
            xs = build.random_placement(self.box, self.pos, comp.xinit, ntries=ntries)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    #place_bilayer () 脂质双层膜排布
    def place_bilayer(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """
        #print('bilayergrid.shape',self.bilayergrid.shape)
        inserted = False
        while not inserted:
            xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit)
            if not inserted:
                xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit, upward=False)
                idx = np.random.randint(self.bilayergrid.shape[0])
                self.bilayergrid[0] = self.bilayergrid[idx]
                self.bilayergrid = np.delete(self.bilayergrid,idx,axis=0)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

#add_bonds /add_angles/add_restraints /add_exclusions 内部作用辅助函数
    def add_bonds(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_bonds(offset)
        self.add_exclusions(exclusion_map)
    #offset = 当前分子第一个珠子在全局体系的索引偏移；添加分子内共价键，并返回需要关闭非键作用的粒子对，执行排除。

    def add_angles(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_angles(offset)
        self.add_exclusions(exclusion_map)
    #RNA 专用，添加角度弯曲势，同样生成非键排除对。

    def add_restraints(self, comp, offset, exclude_nonbonded = True):
        """ Add restraints to single molecule. """

        exclusion_map = comp.add_restraints(offset)
        if exclude_nonbonded: # exclude ah, yu when restraining
            self.add_exclusions(exclusion_map)
    #single molecule添加 GO 拓扑约束；约束成对粒子关闭 LJ / 静电非键作用避免双重势冲突。

    def add_custom_restraints(self, exclude_nonbonded = True):
        exclusion_map = []
        # self.custom_restr_pairs = []
        self.cres = interactions.init_restraints(self.custom_restraint_type)
        for i, j, r, k in self.custom_restr_abs: # i, j, r, k
            self.cres, restr_pair = interactions.add_single_restraint(
                self.cres, self.custom_restraint_type, r, k, i, j)
            # self.custom_restr_pairs.append(restr_pair)
            exclusion_map.append([i,j])
        if exclude_nonbonded: # exclude cres when restraining
            self.add_exclusions(exclusion_map)

    def add_exclusions(self, exclusion_map):
        # exclude LJ, YU for restrained pairs
        for excl in exclusion_map:
            self.ah = interactions.add_exclusion(self.ah, excl[0], excl[1])    #ah：疏水 + 空间排斥（范德华类作用）
            self.yu = interactions.add_exclusion(self.yu, excl[0], excl[1])    #yu：盐溶液中屏蔽静电库仑作用
            if self.nlipids > 0 or self.ncookelipids > 0:
                self.cos.addExclusion(excl[0], excl[1])
                self.cn.addExclusion(excl[0], excl[1])
#对约束 / 成键粒子对，在所有非键力场中添加排除，不计算分子内非键相互作用。

#add_interactions () 单分子完整力场挂载逻辑
    def add_interactions(self,comp):
        """
        Protein interactions for one molecule of composition comp
        """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system

        # Add Ashbaugh-Hatch
        for sig, lam in zip(comp.sigmas, comp.lambdas):
            if comp.molecule_type in ['lipid', 'cooke_lipid']:
                self.ah.addParticle([sig*unit.nanometer, lam, 0])
            elif comp.molecule_type == 'crowder':
                self.ah.addParticle([sig*unit.nanometer, lam, -1])
            else: # protein, RNA
                self.ah.addParticle([sig*unit.nanometer, lam, 1])
            #给每个珠子注册 LJ 参数：半径 sig、混合 lambda、分子类型标记（区分蛋白 / 脂质 / 拥挤物交叉作用）。
            if self.nlipids > 0 or self.ncookelipids > 0:
                if comp.molecule_type in ['lipid', 'cooke_lipid']:
                    self.cos.addParticle([sig*unit.nanometer, lam, 0])
                else:
                    self.cos.addParticle([sig*unit.nanometer, lam, 1])
            #膜体系额外注册脂质聚集势参数。
        
        # Add Debye-Huckel
        for q in comp.qs:
            self.yu.addParticle([q])   #每个珠子注册电荷 q 给静电 DH 势。

        # Add Charge-Nonpolar Interaction,电荷 - 脂质疏水交叉作用参数注册。
        if self.nlipids > 0 or self.ncookelipids > 0:
            id_cn = 1 if comp.molecule_type == 'protein' else -1
            for sig, alpha, q in zip(comp.sigmas, comp.alphas, comp.qs):
                self.cn.addParticle([(sig/2)**3, alpha, q, id_cn])
        
       #依次添加键、RNA 角、GO 约束；verbose 开启输出键 / 约束文件用于调试。
        # Add bonds
        self.add_bonds(comp, offset)

        if comp.molecule_type == 'rna':
            self.add_angles(comp, offset)

        # Add restraints
        if comp.restraint:
            self.add_restraints(comp,offset)

        # write lists
        if self.verbose:
            comp.write_bonds(self.path)
            if comp.restraint:
                comp.write_restraints(self.path)

    #add_ext_restraints () 外部全局限制力
    #将分子所有珠子索引注册到自定义外部势场（slab 平衡限制盒子中心）。
    def add_ext_restraints(self,comp):
        """ Add external-potential restraints. """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system
        for i in range(0,comp.nbeads):
            self.rcent.addParticle(i+offset)

   #add_mdtraj_topol () 构建 PDB 拓扑链、残基、原子、键
    def add_mdtraj_topol(self, comp):
        """ Add one molecule to mdtraj topology. """

        # Note: Move this to component eventually.
        chain = self.top.add_chain()

        if comp.molecule_type == 'rna':    #RNA：每个残基生成 P/N 两个珠子，添加链内连续键
            for idx,resname in enumerate(comp.seq):
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom(resname+"P", element=md.element.phosphorus, residue=res)
                self.top.add_atom(resname+"N", element=md.element.nitrogen, residue=res)
            for i in range(comp.nbeads-1):   
                for j in range(1,comp.nbeads):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))
        else:
            for idx,resname in enumerate(comp.seq):
                if comp.molecule_type in ['protein','crowder']:  #蛋白 / 拥挤物：单 CA 粗粒珠子，三字母残基命名，连续肽键。
                    resname = comp.residues.loc[resname,'three']
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom('CA', element=md.element.carbon, residue=res)
            for i in range(chain.n_atoms-1):
                if comp.bond_check(i,i+1):
                    self.top.add_bond(chain.atom(i), chain.atom(i+1))

    def add_particles_system(self,mws):       #add_particles_system () 注册珠子质量到 OpenMM
        """ Add particles of one molecule to openMM system. """

        for mw in mws:                        #mws 质量数组源头就是你这份 residues.csv 参数表的 MW 列
            self.system.addParticle(mw*unit.amu)

    #自定义约束映射 map_custom_restraints /parse_custom_restraints
    #读取的自定义约束文件是.txt文件
    #自定义约束文件和 domain / GO 约束的区别（这条文件独有优势）
#GO 约束：自动根据 PAE 全序列批量生成天然接触，不能手动指定某几对；
#domains.yaml：只能连续残基区间施加质心 / 区间弹簧，无法精确到单个残基点对点；
#自定义 txt：完全手动自选任意两颗珠子（同分子 / 跨蛋白 / 蛋白 - RNA 都可以）定点加弹簧，自由度最高。
    def map_custom_restraints(self):  #读取自定义约束文件（分子名 - 拷贝号 - 珠子号），转换为全局绝对珠子索引，存入 custom_restr_abs。
        """ Map input format for custom restraints to absolute bead number """
        custom_restr = self.parse_custom_restraints(self.fcustom_restraints)
        total_beads = [0]
        for idx, comp in enumerate(self.components):
            comp.start_bead = total_beads[-1]
            total_beads.append(int(comp.nmol * comp.nbeads))
        self.custom_restr_abs = []
        for i,j,r,k in custom_restr:
            print(i,j,r,k)
            crestr = []
            for idx, x in enumerate([i,j]):
                name, copy, bead = x[0], x[1], x[2] # 1-based
                for idx, comp in enumerate(self.components):
                    if comp.name == name:
                        x_abs = comp.start_bead + (copy-1)*comp.nbeads + (bead-1)
                        break
                crestr.append(x_abs)
            crestr.append(float(r))
            crestr.append(float(k))
            self.custom_restr_abs.append(crestr)

    @staticmethod
    def parse_custom_restraints(fcustom_restraints):   #静态方法，读取约束文件，按 | 分割读取两个分子珠子、平衡距离 r、弹簧常数 k，返回解析后的约束列表。
        custom_restraints = []
        with open(fcustom_restraints,'r') as f:
            for line in f.readlines():
                spl = line.split('|')
                i = spl[0].split()
                j = spl[1].split()
                r = spl[2].split()[0]
                k = spl[2].split()[1]
                restr = [
                    [i[0], int(i[1]), int(i[2])],
                    [j[0], int(j[1]), int(j[2])],
                    r,
                    k
                ]
                custom_restraints.append(restr) # 1-based
        return custom_restraints

#simulate () 模拟运行主流程
    def simulate(self):
        """ Simulate. """

        fcheck_in = f'{self.path}/{self.frestart}'
        fcheck_out = f'{self.path}/restart.chk'
        append = False

        if self.restart == 'pdb' and os.path.isfile(fcheck_in):
            pdb = app.pdbfile.PDBFile(fcheck_in)
        else:
            pdb = app.pdbfile.PDBFile(self.pdb_cg)

        # use langevin integrator
        integrator = openmm.openmm.LangevinMiddleIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
        if self.random_number_seed is not None:
            integrator.setRandomNumberSeed(self.random_number_seed)
        print(integrator.getFriction(),integrator.getTemperature())

        # assemble simulation
        platform = openmm.Platform.getPlatformByName(self.platform)
        if self.platform == 'CPU':
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
                platform.setPropertyDefaultValue('DeviceIndex',str(self.gpu_id))
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)
        print('Running on', platform.getName())

        if (os.path.isfile(fcheck_in)) and (self.restart == 'checkpoint'):
            if not os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'):
                raise Exception(f'Did not find {self.path}/{self.sysname:s}.dcd trajectory to append to!')
            append = True
            print(f'Reading check point file {fcheck_in}')
            print(f'Appending trajectory to {self.path}/{self.sysname:s}.dcd')
            print(f'Appending log file to {self.path}/{self.sysname:s}.log')
            simulation.loadCheckpoint(fcheck_in)
        else:
            if self.restart == 'pdb':
                print(f'Reading in system configuration {self.frestart}')
            elif self.restart == 'checkpoint':
                print(f'No checkpoint file {self.frestart} found: Starting from new system configuration')
            elif self.restart is None:
                print('Starting from new system configuration')
            else:
                raise

            if os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'): # backup old dcd if not restarting from checkpoint
                now = datetime.now()
                dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")
                print(f'Backing up existing {self.path}/{self.sysname:s}.dcd to {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
                os.system(f'mv {self.path}/{self.sysname:s}.dcd {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
            print(f'Writing trajectory to new file {self.path}/{self.sysname:s}.dcd')
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.slab_eq:
            print(f"Starting slab equilibration with k_eq == {self.k_eq:.4f} kJ/(mol*nm) for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
            state_final = simulation.context.getState(getPositions=True)
            rep = app.pdbreporter.PDBReporter(f'{self.path}/equilibration_final.pdb',0)
            rep.report(simulation,state_final)
            pdb = app.pdbfile.PDBFile(f'{self.path}/equilibration_final.pdb')

            for index, force in enumerate(self.system.getForces()):
                if isinstance(force, openmm.CustomExternalForce):
                    print(f'Removing external force {index}')
                    self.system.removeForce(index)
                    break
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.box_eq or self.bilayer_eq:
            print(f"Starting pressure equilibration for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
            state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
            rep = app.pdbreporter.PDBReporter(f'{self.path}/equilibration_final.pdb',0)
            rep.report(simulation,state_final)
            pdb = app.pdbfile.PDBFile(f'{self.path}/equilibration_final.pdb')
            topology = pdb.getTopology()
            a, b, c = state_final.getPeriodicBoxVectors()
            topology.setPeriodicBoxVectors(state_final.getPeriodicBoxVectors())
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            if not self.pressure_coupling:
                for index, force in enumerate(self.system.getForces()):
                    if isinstance(force, openmm.openmm.MonteCarloMembraneBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
                    if isinstance(force, openmm.openmm.MonteCarloAnisotropicBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform)
            simulation.context.setPositions(state_final.getPositions())
            simulation.context.setPeriodicBoxVectors(a, b, c)

        # run simulation
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/{self.sysname:s}.dcd',self.wfreq,append=append))
        simulation.reporters.append(app.statedatareporter.StateDataReporter(f'{self.path}/{self.sysname}.log',self.logfreq,
                step=True,speed=True,elapsedTime=True,potentialEnergy=self.report_potential_energy,separator='\t',append=append))

        print("STARTING SIMULATION", flush=True)
        if self.runtime > 0: # in hours
            simulation.runForClockTime(self.runtime*unit.hour, checkpointFile=fcheck_out, checkpointInterval=30*unit.minute)
        else:
            nbatches = 10
            batch = int(self.steps / nbatches)
            for i in tqdm(range(nbatches),mininterval=1):
                simulation.step(batch)
                simulation.saveCheckpoint(fcheck_out)
        simulation.saveCheckpoint(fcheck_out)

        now = datetime.now()
        dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")

        state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
        rep = app.pdbreporter.PDBReporter(f'{self.path}/{self.sysname}_{dt_string}.pdb',0)
        rep.report(simulation,state_final)
        rep = app.pdbreporter.PDBReporter(f'{self.path}/checkpoint.pdb',0)
        rep.report(simulation,state_final)

#入口 run () 函数（程序启动入口）：1.读取配置 yaml 与组分 yaml；2.实例化 Sim 模拟对象；3.构建完整 MD 体系；4.执行模拟；5.返回模拟实例（方便交互式脚本获取结果）
def run(path='.',fconfig='config.yaml',fcomponents='components.yaml'):
    with open(f'{path}/{fconfig}','r') as stream:
        config = safe_load(stream)

    with open(f'{path}/{fcomponents}','r') as stream:
        components = safe_load(stream)

    mysim = Sim(path,config,components)
    mysim.build_system()
    mysim.simulate()
    return mysim
