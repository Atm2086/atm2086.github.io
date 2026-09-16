---
title: InfiniCore总结
top: false
cover: false
toc: true
mathjax: true
date: 2025-12-25 11:49:06
password:
summary:
tags: 算子
categories: InfiniCore & InfiniLM 学习
---

# 相关问题记录

1. `source` 命令相关问题，重点是当场生效

- 对比： `./test.sh` ，这个命令使用时候，首先要求文件有执行权限，一般来说创建的文件是没有执行权限的，所以可能需要chmod一下；其次系统执行时候会启动一个临时进程比如子进程来跑这个脚本，脚本运行完，进程消失，脚本里定义的变量消失。

- `source`，脚本里对环境做的任何修改（比如修改了 PATH 路径、定义了别名）都会直接留在当前的窗口里（一次性的）。

- 如果不想每次都主动source一遍，可以写进Linux的自动加载清单里，比如 `~/.bashrc`（如果用的是bash）；这样每次登录SSH或打开新窗口系统都会自动执行一遍。

2. 使用Slurm任务提交脚本

```bash
#!/bin/bash
#SBATCH --job-name=test_ops_job              # 任务名
#SBATCH --output=output_%j.log           # 标准输出文件（%j 会替换成 job ID）
#SBATCH --error=error_%j.log             # 标准错误输出文件
#SBATCH --partition=nvidia               # 分区名（机器系统默认分区是 nvidia）
#SBATCH --nodes=1                        # 需要的节点数
#SBATCH --ntasks=1                       # 总任务数（通常 = 节点数 × 每节点任务数）
#SBATCH --cpus-per-task=16               # 每个任务需要的 CPU 核心数
#SBATCH --gres=gpu:nvidia:4              # 请求 4 块 GPU（nvidia 是 Gres 类型）
#SBATCH --mem=256G                       # 请求的内存
#SBATCH --time=00:20:00                  # 运行时间上限

# 需要用到计算资源的命令
# 推荐使用 srun 启动主程序，自动绑定资源
srun python scripts/python_test.py --nvidia
```

- `#!/bin/bash` 第一行告诉系统应该用/bin/bash解释器来读它

- `#SBATCH` 一种特殊的伪指令，执行`sbatch`提交命令的时候，Slurm后台程序会去专门扫描文件头来分配资源。

- 这个脚本的运行方式需要注意

    - 即使是.sh脚本文件，系统没有检查到x权限时候也会permission denied，新创建的文本文件基本是默认只能读写

    - **“直接运行脚本”vs“提交给Slurm调度器”**，如果是直接`.sh`的方式相当于在登录节点上跑，消耗登录节点资源，而不是去申请的资源上跑；提交给Slurm调度器，只需要文本即可，`sbatch`负责读取文件内容

    - 什么是登陆节点/计算节点？集群的运行机制，看成前台和后台，前台负责接待（ssh登录、查看文件、编写脚本等等，内存有限，硬件配置弱，肯定不适合干重活），后台不能直接进，必须通过订单（sbatch）传任务

- `slurm`方法中，涉及到一些其他相关方面

    - 检查当前目录剩余空间 `df -h .` disk free

    - `squeue` 查看当前的slurm里任务情况，PD即为pending，任务运行结束后，会在脚本同级目录下生成一个output_jobid.log，如果想要实时看，可以`tail -f output_jobid.log`

    - `scancel job_id` 取消任务

    - 如果显示PartitionTimeLimit，则可能是被拦截了，因为超过了分区的最大允许时长

    - `scontrol show partition nvidia` 查看分区详情

# Infini本体相关问题