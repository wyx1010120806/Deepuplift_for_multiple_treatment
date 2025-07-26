- 项目概述
基于DeepUplift框架，扩展了多treatment训练支持及特征选择、embedding等模块，支持以下模型：
  - T-learner（已实现）
  - S-learner（已实现）
  - Tarnet（已实现）
  - ESN+Tarnet（已实现）
  - DESCN（待实现）
- 使用注意事项
  - 模型初始化：必须传入`treatment_label_list`参数（类型为`list`），用于定义treatment标签。其中：
    - `0`表示控制组
    - `1, 2, ..., n`表示不同treatment组
示例：`treatment_label_list=[0,1,2,3]`
