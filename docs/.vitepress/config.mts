import { defineConfig } from 'vitepress'

export default defineConfig({
  title: '褚成志的昇腾CANN笔记',
  description: '华为昇腾CANN开发知识',
  lang: 'zh-CN',
  appearance: 'dark',

  themeConfig: {
    // 站点标题与 logo
    siteTitle: '褚成志的CANN笔记',

    // 搜索
    search: {
      provider: 'local',
    },

    // 右侧目录
    outline: {
      level: [2, 3],
      label: '本页目录',
    },

    // 导航栏
    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/chapters/00_overview' },
    ],

    // 侧边栏
    sidebar: [
      {
        text: '概览',
        items: [
          { text: 'CANN 生态全景概览', link: '/chapters/00_overview' },
        ],
      },
      {
        text: '硬件基础',
        items: [
          { text: '昇腾 AI 处理器硬件体系', link: '/chapters/01_ascend_hardware' },
          { text: '达芬奇架构深度解析', link: '/chapters/02_davinci_arch' },
        ],
      },
      {
        text: 'CANN 软件栈',
        items: [
          { text: 'CANN 软件栈全景', link: '/chapters/03_cann_stack' },
          { text: '驱动层与 Runtime 运行时', link: '/chapters/04_driver_runtime' },
        ],
      },
      {
        text: 'AscendCL 应用开发',
        items: [
          { text: 'AscendCL 核心开发接口', link: '/chapters/05_ascendcl_core' },
          { text: 'AscendCL 内存管理深度解析', link: '/chapters/06_ascendcl_memory' },
          { text: '模型推理与 DVPP 媒体处理', link: '/chapters/07_ascendcl_model_inference' },
        ],
      },
      {
        text: '算子开发',
        items: [
          { text: '算子开发体系概览', link: '/chapters/08_operator_overview' },
          { text: 'TBE 张量加速引擎', link: '/chapters/09_tbe_dsl' },
          { text: 'Ascend C：新一代算子编程语言', link: '/chapters/10_ascend_c' },
        ],
      },
      {
        text: '图引擎与编译',
        items: [
          { text: 'GE 图引擎：图优化与执行', link: '/chapters/11_ge_graph_engine' },
          { text: 'ATC 模型转换工具', link: '/chapters/12_atc_model_convert' },
        ],
      },
      {
        text: '通信与分布式',
        items: [
          { text: 'HCCL 集合通信库', link: '/chapters/13_hccl' },
          { text: '分布式训练策略', link: '/chapters/14_distributed_training' },
        ],
      },
      {
        text: '框架适配',
        items: [
          { text: 'Framework Adaptor：框架适配层', link: '/chapters/15_framework_adaptor' },
          { text: 'MindSpore 与 PyTorch 在昇腾上的开发实践', link: '/chapters/16_mindspore_pytorch' },
        ],
      },
      {
        text: '调优与工具链',
        items: [
          { text: 'AOE 调优引擎', link: '/chapters/17_aoe_tuning' },
          { text: '性能分析与调试工具', link: '/chapters/18_profiling_tools' },
        ],
      },
      {
        text: '部署与生态',
        items: [
          { text: '昇腾应用部署实践', link: '/chapters/19_deployment' },
          { text: '昇腾生态与未来演进', link: '/chapters/20_ecosystem_roadmap' },
        ],
      },
    ],

    // 页脚
    footer: {
      message: '基于昇腾CANN文档整理',
      copyright: '© 2026 褚成志',
    },

    // 编辑链接文字
    docFooter: {
      prev: '上一页',
      next: '下一页',
    },

    // 深色模式切换文字
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
  },
})
