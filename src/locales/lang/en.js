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
      "report": "Technical Report",
      "download": "Download",
      "github": "GitHub",
      "visualization": "Visualization",
      "about": "About Us"
    }
  },
  "overview": {
    "title": "Overview",
    "paragraphs": {
      "1": "Bimanual manipulation is essential for achieving human-like dexterity in robots, but the large-scale and diverse bimanual robot datasets remain scarce due to hardware heterogeneity across robotic platforms.",
      "2": "To address the challenge, we present RoboCOIN, a comprehensive multi-embodiment bimanual manipulation dataset with over <span style='color: blue;'>180,000 demonstrations</span> collected from <span style='color: blue;'>15 distinct robotic platforms</span>. The dataset covers <span style='color: blue;'>16 scenarios</span> including residential, commercial, working environments, with <span style='color: blue;'>421 tasks</span> systematically organized by bimanual coordination patterns and object properties. Our key innovation is a hierarchical capability pyramid that provides multi-level annotations, spanning trajectory-level concepts, segment-level subtasks, and frame-level kinematics.",
      "3": "We further develop CoRobot, a comprehensive processing framework featuring Robot Trajectory Markup Language (RTML) for quality assessment, automated annotation generation, and unified multi-embodiment management. Extensive experiments demonstrate the reliability and effectiveness of RoboCOIN in multi-embodiment bimanual learning, with significant performance improvements across various model architectures and robotic platforms. The complete dataset and framework will be open-sourced and publicly available for further research purposes."
    },
    "boxes": {
      "robot": "Dual-Arm Embodiments",
      "trajectory": "Real-world Trajectories",
      "scenario": "Diverse Scenarios",
      "task": "Comprehensive Tasks"
    }
  },
  "highlights": {
    "title": "Highlights",
    "robot": {
      "title": "Diverse Dual-Arm Robotic Platforms",
      "desc": "Includes 15 different robot models across three categories: dual-arm, semi-humanoid, and humanoid robots.",
      "dual": "Dual-Arm",
      "half_humanoid": "Half-Humanoid",
      "humanoid": "Humanoid",
    },
    "scenario": {
      "title": "Varied and Realistic Scenarios",
      "desc": "Covers 16 different scenarios across residential, commercial, and office environments.",
      "residential": "Residential",
      "commercial": "Commercial",
      "working": "Working",
      "bedroom": "Bedroom",
      "living_room": "Living-Room",
      "children_room": "Children-Room",
      "home": "Home",
      "kitchen": "Kitchen",
      "restaurant": "Restaurant",
      "courier_station": "Courier Station",
      "supermarket": "Supermarket",
      "amusement_park": "Amusement Park",
      "cafe": "Cafe",
      "factory": "Factory",
      "warehouse": "Warehouse",
      "classroom": "Classroom",
      "laboratory": "Laboratory",
      "laundry": "Laundry",
      "toll_booth": "Toll Booth",
      "office": "Office",
      "school": "School",
      "hospital": "Hospital",
    },
    "swiper": {
      "title": "Robot Introduction",
      "dual": "Dual-Arm",
      "half_humanoid": "Half-Humanoid",
      "humanoid": "Humanoid",
      "dexterous": "Dexterous Hand",
    },
    "scenario_swiper": {
      "title": "Scenario Introduction",
      "residential": "Residential",
      "commercial": "Commercial",
      "working": "Working",
      "bed": "Make Bed",
      "cloth": "Fold Clothes",
      "kitchen": "Microwave Meal",
      "food": "Prepare Food",
      "fruit": "Place Fruits",
      "icecream": "Make Ice Cream",
      "recycle": "Recycle Rubbish",
      "supermarket": "Sell Goods",
      "toilet": "Clean Toilet",
      "pipeline": "Line Work",
    },
    "task": {
      "title": "Multi-Dimensional Task Taxonomy",
      "desc": "Tasks are organized in a hierarchical grid based on motion coordination and object variability.",
      "coordination": {
        "low": "Low Action Coordination",
        "high": "High Action Coordination",
      },
      "object": {
        "rigid": "Rigid Objects",
        "hinged": "Hinged Objects",
        "deformable": "Deformable Objects"
      }
    },
    "demos": {
      "coordination": {
          "low": "Low Action\nCoordination",
          "high": "High Action\nCoordination"
      },
      "object": {
          "rigid": "Rigid\nObjects",
          "hinged": "Hinged\nObjects",
          "deformable": "Deformable\nObjects"
      },
      "play": "Click to play",
      "task": {
        "carry_parts": "Carry Parts",
        "open_washer": "Open Washer",
        "flip_book": "Flip Book",
        "arrange_books": "Arrange Books",
        "heat_food": "Heat Food",
        "pass_pack": "Pass Pack"
      }
    },
    "pyramid": {
      "title": "Hierarchical Capability Pyramid",
      "desc": "Provides scene-level, segment-level, and frame-level annotations.",
      "frame": "Frame",
      "levels": {
        "trajectory": "Trajectory-Level Annotation",
        "segment": "Segment-Level Annotation",
        "frame": "Frame-Level Annotations"
      }
    }
  },
  "comparison": {
    "title": "Comparsion with Existing Robotic Datasets",
    "desc": "Comparison of existing real-world datasets for robot manipulation. All data is drawn from the original paper or RoboMIND paper. <br/><span style='color: gray;'>†not a dataset in itself, but an aggregation of existing datasets.</span>",
    "table": {
      "header": {
        "dataset": "Dataset",
        "arm": "Arm",
        "embodiment": "Embodiments",
        "trajectory": "Trajectories",
        "task": "Tasks",
        "skill": "Skills",
        "dexterous": "Dexterous",
        "annotation": "Annotation",
        "collection": "Collection Method"
      },
      "content": {
        "dual": "Dual",
        "single": "Single",
        "single_dual": "Single & Dual",
        "na": "N/A",
        "no": "No",
        "flat": "Flat",
        "hierarchical": "Hierarchical",
        "scripted": "Scripted",
        "human_teleoperation": "Human Teleoperation",
        "30human_70scripted": "30% Human / 70% Scripted",
        "85human_15scripted": "85% Human / 15% Scripted",
        "dataset_aggregation": "Dataset Aggregation",
      }
    }
  },
  "corobot": {
    "title": "CoRobot Framework",
    "desc": "CoRobot is a comprehensive data processing and management framework designed for robotic applications, enabling efficient data collection, conversion, annotation, storage, and model training.",
    "modules": {
      "data_collect": "Supports data collection and teleoperation for 15 robot platforms",
      "data_convert": "Supports conversion of mainstream robot data formats such as RLDS, HDF5, LeRobot, etc.",
      "data_forge": "Data post-processing and annotation module that detects issues like stuttering, static frames, and timestamp misalignment, providing scene-level, task-level, and description-level annotations",
      "robocoin": "A large-scale dual-arm robot dataset covering 15 robot bodies and over 180,000 trajectories from 421 tasks across 16 scenarios",
      "data_manage": "Atomic data storage and management tool supporting data retrieval, visualization, extraction, and download",
      "data_train": "Model training module for various robot learning algorithms, supporting the integration of embodied models like OpenPI, RDT, and ACT, and catering to customized model training requirements"
    }
  },
  "collaborating": {
    "title": "Collaborating Institutions",
    "desc": "We are proud to collaborate with leading institutions in the field of robotics.",
    "institutions": {
      "agilex": "AgileX Robotics",
      "ai2": "AI² Robotics",
      "galaxea": "Galaxea",
      "galbot": "Galbot",
      "leju": "Leju Robotics",
      "ant": "Ant Digital Technologies, Ant Group",
      "realman": "Realman Robotics",
      "tianqing": "TQ-Artisan",
      "uestc": "University of Electronic Science and Technology of China",
      "bupt": "Beijing University of Posts and Telecommunications",
      "ruc": "Renmin University of China",
      "hust": "Huazhong University of Science and Technology",
      "pku": "Peking University",
      "thu": "Tsinghua University",
      "stanford": "Stanford University",
      "berkeley": "University of California, Berkeley",
      "booster": "Booster Robotics",
      "dora": "Dora Community",
    }
  },
  "footer": {
    "title": "Official Open Source in Nov 2025",
    "boxes": {
      "robot": "15 Dual-arm Robots",
      "data": "180K+ Real World Trajectories",
      "toolchain": "Full Toolchain"
    },
    "start": "Get Started Now:",
    "text": "Scan the QR code to join the group, get priority notification on the open source day, and have the opportunity to contact the units involved in the construction of this dataset in advance.",
    "copyright": "© 2025 Beijing Academy of Artificial Intelligence. All rights reserved."
  }
}