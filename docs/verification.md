# 验证范围

本地验证针对当前导出副本，真实供应商调用未执行。依赖安装与浏览器测试会在审核记录中保留结果。全新 Git clone 的安装、CLI、网页启动、合成样例提交、人工审核、重启持久化和离线测试为本次验收路径。

源码包保留运行模块回归测试，不包含研究语料、采集器、内部 owner 活动及相关测试。不是原仓库全套测试或企业效果验收。

当前未验证：特定 Agent 宿主配置、真实 Jev／Ark 调用、其他操作系统与 Python 版本矩阵、持续监听、外部执行回执、H3C 专项效果。没有把历史样本数量作为当前效果指标。

打包时复制了当前工作区，包括尚未提交的网页改动；不携带原仓库历史。程序静态资源见 [HTML](../src/jevtest/static/index.html)、[CSS](../src/jevtest/static/review.css)、[JS](../src/jevtest/static/review.js) 和 [favicon](../src/jevtest/static/favicon.svg)。

本次隔离运行：286 项通过，14 项 live 测试未执行；网页合成样例提交、defer→correct→accept 已通过，390px 视口没有水平溢出。CLI 环境样例与合成 smoke 已分别核验。

已知 UI 问题：offline 初始化没有更新顶部连接徽标，但接口与提交可正常使用。以 /v1/health 与详情中运行模式为准。应用源码未为展示改动。
