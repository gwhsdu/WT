import numpy as np
import torch
from io import StringIO
import pandas as pd
import os
import shutil
from .utils import mesh_convertor, numpy2string


def readonestep(stepdir):
    with open(os.path.join(stepdir, 'p'), 'r') as f:
        num_mesh = int(f.readlines()[20])
    with open(os.path.join(stepdir, 'p'), 'r') as f:
        p = pd.read_csv(f, skiprows=21, nrows=num_mesh, header=0, delim_whitespace=True, dtype=np.float32).to_numpy()

    with open(os.path.join(stepdir, 'U'), 'r') as f:
        contents = f.readlines()

    content = [i[1:-3] for i in contents[22:]]
    ff = StringIO('\n'.join(content))
    u = pd.read_csv(ff, nrows=num_mesh, header=None, delim_whitespace=True, dtype=np.float32).to_numpy()
    return p, u


def readall(dir):
    presults = []
    uresults = []
    times = os.listdir(dir)
    os.chdir(dir)
    try:
        times.remove('0')
    except:
        pass
    for i in times:
        try:
            float(i)
        except:
            times.remove(i)
    timeID = np.array(times, dtype=float)
    timeID = timeID.argsort()
    for t in timeID:
        print(times[t])
        p, u = readonestep(times[t])
        presults.append(p)
        uresults.append(u)
        # print('step: {0} read'.format(i))

    return np.stack(presults, axis=0), np.stack(uresults, axis=0)


def map2coarse(src, dst, t, len_stp=0.001):
    tmpdir = 'coarse_tmp'
    oldpath = os.getcwd()
    try:
        shutil.copytree(src, tmpdir)
    except FileExistsError:
        pass
    os.chdir(dst)
    print('map field {0} ...'.format(t))

    assert os.system(
        'mapFields {0} -consistent -sourceTime {2} -case {1} >> /dev/null' \
            .format(src, tmpdir, t)) == 0

    os.chdir(tmpdir)

    print('icoFoam solving {0} ...'.format(t))
    assert os.system('icoFoam >> /dev/null') == 0
    print('icoFoam time {0} finished!'.format(t))

    os.chdir(oldpath)
    shutil.move(os.path.join(tmpdir, str(len_stp)), os.path.join(dst, '{0:.3f}'.format(t + len_stp)))


def uheader(n):
    return """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>\n""" + \
        str(n) + '\n' + \
        '(\n'


def utail(pos, t, num_bcpoints=25):
    def ubc(num_mesh, pos, t):
        x = (np.linspace(0, num_mesh - 1, num_mesh) + 0.5) / num_mesh
        u = np.exp(-50 * (x - pos) * (x - pos))         # 边界信息
        v = np.sin(t) * np.exp(-50 * (x - pos) * (x - pos))    # 边界信息
        return np.stack((u, v, np.zeros_like(u)), axis=1)

    bcvalue = numpy2string(ubc(num_bcpoints, pos, t))
    return """\
)
;

boundaryField
{
    walls
    {
        type            noSlip;
    }
    inlet
    {
        type            groovyBC;
        refValue        nonuniform List<vector>\n""" + \
        str(num_bcpoints) + '\n' + \
        '(\n' + \
        bcvalue + \
        '\n)\n;\n' + \
        """\
                refGradient     uniform (0 0 0);
                valueFraction   uniform 1;
                value           nonuniform List<vector>\n""" + \
        str(num_bcpoints) + '\n' + \
        '(\n' + \
        bcvalue + \
        '\n)\n;\n' + \
        """\
                valueExpression "vector(exp(-50*(pos().y-{0})*(pos().y-{0})),sin(time())*pos().y/pos().y*exp(-50*(pos().y-{0})*(pos().y-{0})),0)";""".format(
            pos) + \
        """
                gradientExpression "vector(0,0,0)";
                fractionExpression "1";
                evaluateDuringConstruction 0;
                cyclicSlave     0;
                variables       "";
                timelines       (
        );
                lookuptables    (
        );
                lookuptables2D  (
        );
            }
            outlet
            {
                type            zeroGradient;
            }
            frontAndBack
            {
                type            empty;
            }
        }
        
        
        // ************************************************************************* //"""


def pheader(n):
    return """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   nonuniform List<scalar>\n""" + \
        str(n) + '\n' + \
        '(\n'


ptail = """\
)
;

boundaryField
{
    walls
    {
        type            zeroGradient;
    }
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    frontAndBack
    {
        type            empty;
    }
}


// ************************************************************************* //
"""


def viscosity(mu, m=None):
    transportProperties = """\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  8
     \\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel Newtonian ;\n""" + \
                          'nu              nu [0 2 -1 0 0 0 0] {0:.5g};\n'.format(mu)
    if m != None: transportProperties += 'm               [0 0 -1 0 0 0 0] {0:.3g};\n\n'.format(m)
    transportProperties += """
// ************************************************************************* //
"""
    return transportProperties


def writeofvec(u, dir, pos, t, inletpoints):
    num_mesh = u.shape[0]

    u = np.concatenate([u, np.zeros((num_mesh, 1))], axis=1)

    ustring = numpy2string(u)

    towrite = uheader(num_mesh) + ustring + utail(pos, t, inletpoints)

    with open(os.path.join(dir, 'U'), 'w+') as f:
        f.write(towrite)


def writeofsca(p, dir):
    num_mesh = p.shape[0]

    pstring = numpy2string(p, '%.6g')

    towrite = pheader(num_mesh) + pstring + ptail

    with open(os.path.join(dir, 'p'), 'w+') as f:
        f.write(towrite)


def writeofvis(mu, dir, m=None):
    towrite = viscosity(mu, m)
    with open(os.path.join(dir, 'transportProperties'), 'w+') as f:
        f.write(towrite)


class OneStepRunOFCoarse(object):  # 旨在与 OpenFOAM 模拟软件进行交互，运行一个模拟步骤并提取结果，主要用于流体动力学的粗网格案例
    def __init__(self, template_path, tmp_path, dt, cmesh,
                 Ct1: float, Ct2: float, num_inletpoints: float,
                 solver='icoFoam', m=0.08) -> None:
        super().__init__()        # 这里的template_path模板目录的路径，包含 OpenFOAM 配置文件。     类的构造函数，用来初始化对象，__init__ 方法会在类的实例化时自动执行，并且它只会执行一次
        try:
            shutil.copytree(template_path, tmp_path)
        except FileExistsError:
            pass
        self.tmp_path = tmp_path  # 模拟执行的工作目录路径
        self.dt = dt  # 模拟的时间步长
        self.cmesh = cmesh  # 计算网格（可能是一个表示网格维度的元组)
        self.Ct1 = Ct1  # 涡轮机1的推力系数
        self.Ct2 = Ct2  # 涡轮机2的推力系数
        self.num_inletpoints = num_inletpoints  # 入口的点数，用于指定边界条件 。 这里给定的是粗网格上x方向的剖分数cmesh[0]
        self.solver = solver
        if solver == 'icoFoam': m = None   # OpenFOAM 求解器（默认为 'icoFoam'，将参数 m 设置为 None（因为在这种情况下 m 可能不需要）
        writeofvis(self.Ct1, self.Ct2, os.path.join(self.tmp_path, 'constant'), m)  # 将物理属性 self.Ct1，self.Ct2 和参数 m 写入模拟的 constant 目录
        with open(os.path.join(self.tmp_path, 'system/controlDict'), 'r') as f:
            self.controlDict = f.readlines()  # 打开 OpenFOAM 的 controlDict 文件并读取其内容

    def __call__(self, u0: torch.Tensor, t) -> torch.Tensor:   #当你创建了类的实例之后，可以像调用普通函数一样调用这个实例

        error = False

        # p = u0[0, 2].reshape(1, -1).permute(1, 0)
        # u = u0[0, :2].reshape(2, -1).permute(1, 0)
        p = u0[0, 3].reshape(1, -1).permute(1, 0)
        u = u0[0, :3].reshape(3, -1).permute(1, 0)

        t0s = '{0:3g}'.format(t).replace(' ', '')              # 当前步时间
        t1s = '{0:3g}'.format(t + self.dt).replace(' ', '')    # 下一个时间步时间

        U0path = os.path.join(self.tmp_path, t0s)  # 当前时间步的文件夹路径
        U1path = os.path.join(self.tmp_path, t1s)  # 下一个时间步的文件夹路径

        if not os.path.exists(U0path): os.makedirs(U0path)  # 检查 U0path 是否存在。如果不存在，则创建该目录。

        # writeofvec(u.numpy(), U0path, self.Ct1, t, self.num_inletpoints)
        writeofvec(u.numpy(), U0path, self.Ct1,self.Ct2, t, self.num_inletpoints)
        writeofsca(p.numpy(), U0path)

        # write controlDict
        self.controlDict[21] = 'startTime\t{0};\n'.format(t0s)  # 修改开始时间
        self.controlDict[25] = 'endTime\t{0};\n'.format(t1s)    # 修改结束时间
        with open(os.path.join(self.tmp_path, 'system/controlDict'), 'w') as f:
            f.writelines(self.controlDict)   # 保存修改后的 controlDict

        oldpath = os.getcwd()   # 获取当前的工作目录，并将其存储在 oldpath 变量中
        os.chdir(self.tmp_path)   # 切换当前工作目录到 self.tmp_path
        try:
            ofreturn = os.system('{0} > Foam.log 2>&1'.format(self.solver))   # 执行 OpenFOAM 求解器的命令，并将标准输出（stdout）和标准错误输出（stderr）都重定向到 Foam.log 文件中
        except:
            error = True  # 设置 error = True，表示发生了错误
        if ofreturn != 0:  # 返回值为 0 表示命令成功执行，非 0 表示执行失败
            error = True
        os.chdir(oldpath)  # 恢复到之前保存的原始工作目录
        if error:
            return float('nan') * torch.ones_like(u0), error   # 如果求解器执行失败，则返回一个全是 NaN 的张量
        else:
            p, u = readonestep(U1path)  # 如果求解器执行成功，则通过 readonestep(U1path) 读取仿真结果
            p = torch.from_numpy(p).float()
            u = torch.from_numpy(u).float()
            p = p.reshape(1, 1, *self.cmesh)  # 调整压力场 p 的形状
            # u = u.permute(1, 0).reshape(2, *self.cmesh).unsqueeze(0)
            u = u.permute(1, 0).reshape(3, *self.cmesh).unsqueeze(0)  # 调整速度场 u 的形状
            return torch.cat([u, p], dim=1), error   # 拼接速度场 u 和压力场 p ， 返回拼接后的张量和 error 标志


if __name__ == '__main__':

    res = [2, 4, 6, 8, 10]
    ps = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    mscvter = mesh_convertor((100, 400), (25, 100), dim=2, align_corners=False)
    fdata = torch.load('/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/pipefine.pt'
                       ).float()
    result = []
    for Ct1 in range(len(ps)):
        p_result = []
        for re in range(len(res)):
            csolver = OneStepRunOFCoarse('/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/test',
                                         '/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/tmp', 0.8, (25, 100),
                                         ps[Ct1], 0.001 / res[re]
                                         , 25)
            r_result = []
            if Ct1 == 8 and re == 4:
                continue
            phy_time = 15.2
            for t in range(56):
                u, error = csolver(mscvter.down(fdata[Ct1 * len(res) + re, t:t + 1]), phy_time)
                phy_time += 0.8
                u = mscvter.up(u)
                r_result.append(u)
                if error:
                    print('{0} Ct1,{1} Re,{2} time, error'.format(ps[Ct1], res[re], phy_time))
                    raise Exception('error')
            r_result = torch.cat(r_result, dim=0)
            p_result.append(r_result)
            print('{0} Ct1,{1} Re, done'.format(ps[Ct1], res[re]))
        p_result = torch.stack(p_result, dim=0)
        result.append(p_result)
    result = torch.cat(result, dim=0)
    print(result.shape)
    torch.save(result, '/home/lxy/store/projects/dynamic/PDE_structure/OpenFoam/pipeflow_more_coarse.pt')
