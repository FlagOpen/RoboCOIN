export default {
  "stats": {
    "num_embodiments": "15",
    "num_trajectories": "180K+",
    "num_scenarios": "16",
    "num_tasks": "421",
    "num_actions": "36",
    "num_objects": "432",
  },
  "header": {
    "navs": {
      "report": "技术报告",
      "download": "下载",
      "github": "GitHub",
      "visualization": "数据可视化", 
      "about": "关于我们"
    }
  },
  "overview": {
    "title": "概览",
    "paragraphs": {
      "1": "实现类人灵巧性的机器人双手操作至关重要，但由于不同机器人平台间的硬件异构性，大规模、多样化的双手操作机器人数据集仍然稀缺。",
      "2": "为应对这一挑战，我们推出了RoboCOIN，这是一个全面的多形态双手操作数据集，包含从<span style='color: blue;'>15个不同机器人平台</span>收集的超过<span style='color: blue;'>180,000次演示</span>。该数据集涵盖居住、商业、工作环境等<span style='color: blue;'>16个场景</span>，包含<span style='color: blue;'>421项任务</span>，这些任务根据双手协调模式和物体属性进行了系统化组织。我们的核心创新在于一个分层能力金字塔，它提供了多层次标注，涵盖轨迹级概念、分段级子任务和帧级运动学信息。",
      "3": "我们进一步开发了CoRobot，一个综合性处理框架，其特色是机器人轨迹标记语言（RTML），用于质量评估、自动化标注生成和统一的多形态平台管理。大量实验证明了RoboCOIN在多形态双手操作学习中的可靠性和有效性，在各种模型架构和机器人平台上均显示出显著的性能提升。完整的数据集和框架将开源并公开提供，以供进一步研究之用。"
    },
    "boxes": {
      "robot": "双臂机器人平台",
      "trajectory": "真实世界轨迹", 
      "scenario": "多样化场景",
      "task": "全面任务覆盖"
    }
  },
  "highlights": {
    "title": "核心亮点",
    "robot": {
      "title": "多样化的双臂机器人平台",
      "desc": "包含双臂型、半人形、人形在内的共15种不同的机器人模型。",
      "dual": "双臂机器人",
      "half_humanoid": "半人形机器人",
      "humanoid": "人形机器人",
    },
    "scenario": {
      "title": "多样化的真实场景",
      "desc": "涵盖住宅、商业和办公环境的16种不同场景。",
      "residential": "住宅区",
      "commercial": "商业区",
      "working": "工作区",
      "bedroom": "卧室",
      "living_room": "客厅",
      "children_room": "儿童房",
      "home": "家庭",
      "kitchen": "厨房",
      "restaurant": "餐厅",
      "courier_station": "快递站",
      "supermarket": "超市",
      "amusement_park": "游乐园",
      "cafe": "咖啡馆",
      "factory": "工厂",
      "warehouse": "仓库",
      "classroom": "教室",
      "laboratory": "实验室",
      "laundry": "洗衣房",
      "toll_booth": "收费站",
      "office": "办公室",
      "school": "学校",
      "hospital": "医院"
    },
    "swiper": {
      "title": "机器人类型介绍",
      "dual": "双臂机器人",
      "half_humanoid": "半人形机器人",
      "humanoid": "人形机器人",
      "dexterous": "灵巧手"
    },
    "scenario_swiper": {
      "title": "场景介绍",
      "residential": "住宅场景",
      "commercial": "商业场景", 
      "working": "工作场景",
      "bed": "整理床铺",
      "cloth": "折叠衣物",
      "kitchen": "微波炉加热",
      "food": "准备食物",
      "fruit": "摆放水果",
      "icecream": "制作冰淇淋",
      "recycle": "垃圾分类",
      "supermarket": "商品售卖",
      "toilet": "清洁马桶",
      "pipeline": "流水线作业"
    },
    "task": {
      "title": "多维度任务分类体系",
      "desc": "任务根据运动协调性和对象可变性，按分层网格形式进行组织。",
      "coordination": {
        "low": "低动作协调性",
        "high": "高动作协调性"
      },
      "object": {
        "rigid": "刚性物体",
        "hinged": "铰链物体",
        "deformable": "可变形物体"
      }
    },
    "demos": {
      "coordination": {
        "low": "低动作\n协调性",
        "high": "高动作\n协调性",
      },
      "object": {
        "rigid": "刚性物体",
        "hinged": "铰链物体",
        "deformable": "可变形物体"
      },
      "play": "点击播放",
      "task": {
        "carry_parts": "搬运零件",
        "open_washer": "打开洗衣机",
        "flip_book": "翻动书籍",
        "arrange_books": "整理书籍",
        "heat_food": "加热食物",
        "pass_pack": "传递包裹"
      }
    },
    "pyramid": {
      "title": "分层能力金字塔",
      "desc": "提供场景级、片段级和帧级注释。",
      "frame": "帧",
      "levels": {
        "trajectory": "轨迹级注释",
        "segment": "片段级注释",
        "frame": "帧级注释"
      }
    }
  },
  "comparison": {
    "title": "与现有机器人数据集的对比",
    "desc": "现有机器人操作真实世界数据集的对比。所有数据均摘自原始论文或RoboMIND论文。<br/><span style='color: gray;'>†其本身并非数据集，而是对现有数据集的聚合。</span>",
    "table": {
      "header": {
        "dataset": "数据集",
        "arm": "机械臂构型",
        "embodiment": "本体平台数量",
        "trajectory": "轨迹数量",
        "task": "任务数量", 
        "skill": "技能数量",
        "dexterous": "灵巧操作",
        "annotation": "标注类型",
        "collection": "采集方式"
      },
      "content": {
        "dual": "双臂",
        "single": "单臂",
        "single_dual": "单双腕混合",
        "na": "不适用",
        "no": "否",
        "flat": "扁平",
        "hierarchical": "层级化",
        "scripted": "脚本录制",
        "human_teleoperation": "人工遥操作",
        "30human_70scripted": "30%人工遥操作 / 70%脚本",
        "85human_15scripted": "85%人工遥操作 / 15%脚本",
        "dataset_aggregation": "数据集聚合"
      }
    }
  },
  "corobot": {
    "title": "CoRobot框架",
    "desc": "CoRobot是一个为机器人应用设计的综合性数据处理和管理框架，实现了高效的数据采集、转换、标注、存储和模型训练。",
    "modules": {
      "data_collect": "支持15款机器人平台的数据采集和遥操作",
      "data_convert": "支持主流机器人数据格式RLDS、HDF5、LeRobot等格式的转换", 
      "data_forge": "数据后处理和标注模块，检测卡顿、静止帧、时间戳错位等信息，提供场景、任务、描述信息标注",
      "robocoin": "大型双臂机器人数据集，涵盖15款本体和18万以上轨迹，来自16个场景的421个任务",
      "data_manage": "数据原子化存储与管理工具，支持数据检索与可视化、抽取、下载",
      "data_train": "多种机器人学习算法的模型训练模块，支持OpenPI、RDT、ACT等具身模型接入，定制化模型训练需求"
    }
  },
  "collaborating": {
    "title": "合作机构",
    "desc": "我们很自豪能与机器人领域的顶尖机构合作。",
    "institutions": {
      "agilex": "松灵机器人",
      "ai2": "智平方机器人",
      "galaxea": "星海图",
      "galbot": "银河通用",
      "leju": "乐聚机器人",
      "ant": "蚂蚁数科，蚂蚁集团",
      "realman": "睿尔曼智能",
      "tianqing": "软通天擎",
      "uestc": "电子科技大学",
      "bupt": "北京邮电大学",
      "ruc": "中国人民大学",
      "hust": "华中科技大学",
      "swjtu": "西南交通大学",
      "pku": "北京大学",
      "thu": "清华大学",
      "stanford": "斯坦福大学",
      "berkeley": "加州大学伯克利分校",
      "booster": "加速进化",
      "dora": "DORA社区"
    }
  },
  "footer": {
    "title": "2025年11月正式开源",
    "boxes": {
      "robot": "15种双臂机器人",
      "data": "18万+真实世界轨迹",
      "toolchain": "完整工具链"
    },
    "start": "即刻尝试：",
    "text": "扫描二维码加入群聊，开源日获取优先通知，并有机会提前接触参与本数据集建设的各单位。",
    "copyright": "© 2025 北京智源人工智能研究院。保留所有权利。"
  }
}