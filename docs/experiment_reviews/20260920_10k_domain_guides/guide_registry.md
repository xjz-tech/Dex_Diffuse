# 新10k guide登记表

五组固定训练seed42，从头训练200epoch，共3600次更新；每组10000步数据=9000训练+1000验证。下表为实际保留训练数据统计，按采样窗口计权。各片段同为250步，因此窗口权重相同。

| 模型 | 实际训练质量范围/g | 质量均值/g | 实际训练摩擦范围 | 摩擦均值 | 数据构成 |
|---|---:|---:|---:|---:|---|
| 轻重量／低摩擦 | 40.357–75.739 | 61.260 | 0.507382–1.428049 | 0.983427 | 本域100% |
| 轻重量／高摩擦 | 40.357–75.739 | 61.260 | 2.511073–3.892074 | 3.225140 | 本域100% |
| 重重量／低摩擦 | 181.072–287.217 | 243.781 | 0.507382–1.428049 | 0.983427 | 本域100% |
| 重重量／高摩擦 | 181.072–287.217 | 243.781 | 2.511073–3.892074 | 3.225140 | 本域100% |
| 四域均衡 | 41.237–287.217 | 153.706 | 0.651158–3.797046 | 2.085060 | 四域各25% |

全部数据来自四域都成功的40个共同初始化模板。均衡组每域9个训练片段、1个验证片段。成功筛选会改变原始均匀采样分布；表中没有把设定采样区间冒充实际数据分布。每片段完整标签和分配见各数据集manifest。

## light_low — 轻重量／低摩擦

- [数据分布与逐片段标签](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/domain10k_20260920_light_low/distribution_manifest.json)
- [训练配置](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_light_low_seed42/training_config.yaml)
- [模型checkpoint](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_light_low_seed42/checkpoints/latest.ckpt)
- checkpoint SHA256：`31ae8b873aa6cf0b74f4b58cd2f06b214f42b24b1f127a3b58f708f9c8e0c7e2`
- 分布manifest SHA256：`f7fb0da81e8eb23e2929605ee9df4ccca3719838c470b7b90aadd0ba2529fb2f`

## light_high — 轻重量／高摩擦

- [数据分布与逐片段标签](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/domain10k_20260920_light_high/distribution_manifest.json)
- [训练配置](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_light_high_seed42/training_config.yaml)
- [模型checkpoint](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_light_high_seed42/checkpoints/latest.ckpt)
- checkpoint SHA256：`2de000aac90a0aaf1a0df3a14f3818c27a9aec28bdfe6bfb275896efa83e8d74`
- 分布manifest SHA256：`090557f4a0bda3d2c8a064b399e402a79f79d70ba3d0f1cdaf93d5083a4a1f25`

## heavy_low — 重重量／低摩擦

- [数据分布与逐片段标签](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/domain10k_20260920_heavy_low/distribution_manifest.json)
- [训练配置](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_heavy_low_seed42/training_config.yaml)
- [模型checkpoint](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_heavy_low_seed42/checkpoints/latest.ckpt)
- checkpoint SHA256：`9f90b92084251b17960b416f4623a27ff8595baa49e356e51029680375d59949`
- 分布manifest SHA256：`6a62d547dc5c51c956a51070518dd517424b7c9e744a3a94da7ade4d75d0b484`

## heavy_high — 重重量／高摩擦

- [数据分布与逐片段标签](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/domain10k_20260920_heavy_high/distribution_manifest.json)
- [训练配置](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_heavy_high_seed42/training_config.yaml)
- [模型checkpoint](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_heavy_high_seed42/checkpoints/latest.ckpt)
- checkpoint SHA256：`0425cd688b544fd9c767aeb0560b0f311dd455bf718d91df7c9d4150cee916a1`
- 分布manifest SHA256：`572141047f7c7e8e439be3e9f43a5c0e05dea3a53f78226ced7400f935a65910`

## balanced — 四域均衡

- [数据分布与逐片段标签](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/data/domain10k_20260920_balanced/distribution_manifest.json)
- [训练配置](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_balanced_seed42/training_config.yaml)
- [模型checkpoint](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/domain10k_20260920_balanced_seed42/checkpoints/latest.ckpt)
- checkpoint SHA256：`8f0ee71aa387719031477238445fe441edf8aef76b711d95385c4dfb66327f48`
- 分布manifest SHA256：`2fec6e2db741fdaf831b01ce704c3c82bb2a683aceb58179dab20bd4f28991f9`

训练模型输入只有关节角、上一目标、目标残差，不输入质量或摩擦标签。验证loss来自各自分布，不作为跨模型排名依据。当前仅一个训练seed，比较应看对应成对仿真结果。