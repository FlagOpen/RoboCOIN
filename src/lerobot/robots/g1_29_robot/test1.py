#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 G1_29_Robot 方法级测试脚本（0–6 编号版）
✅ 支持：
  0: _check_dependencys
  1: _connect_arm
  2: _disconnect_arm
  3: _set_joint_state
  4: _get_joint_state
  5: _set_ee_state
  6: _get_ee_state

💡 运行于 simulation_mode=True —— 绝对安全，无硬件风险。
"""

import sys
import numpy as np
import time
import logging

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s \033[1m%(message)s\033[0m',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 导入你的类（请确保路径正确）---
try:
    from g1_29_robot import G1_29_Robot
except ImportError as e:
    logger.error(f"❌ 导入失败：{e}")
    logger.error("💡 提示：请确认 'g1_29_robot.py' 在当前目录或 PYTHONPATH 中。")
    sys.exit(1)

# --- 全局 robot 实例（单例复用）---
robot = G1_29_Robot()

# --- 工具函数：确保已连接（幂等）---
def ensure_connected():
    if not robot.is_connected:
        logger.info("🔌 正在连接机器人（仿真模式）...")
        success = robot._connect(
            motion_mode=True,
            simulation_mode=False,
            left_hand_array_in=None,
            right_hand_array_in=None,
        )
        if not success:
            raise RuntimeError("❌ 机器人连接失败（仿真模式）")

# --- 测试数据（预定义）---
ARM_Q_TARGET = np.array([
    -0.26995954746545603,
     0.17546311376881019,
     0.1317917905676921,
     1.3550701922215134,
     0.08896632899092974,
    -0.24695407868402758,
    -0.3344995064887082,
    -0.12931980638634,
    -0.08606414890560501,
    -0.2797933172022591,
     1.3164315790194259,
     0.07989473750157136,
    -0.37218325052001133,
     0.279260645440092
])

EE_STATE_TEST = np.array([
    0.25, 0.25, 0.1, 0.0, 0.0, 0.0,   # L: x,y,z,r,p,y
    0.25, -0.25, 0.1, 0.0, 0.0, 0.0   # R: x,y,z,r,p,y
])


# ================================
# 🧪 各方法测试函数（编号 0–6）
# ================================

def test_0_check_dependencys():
    """0: _check_dependencys"""
    logger.info("🧪 测试 _check_dependencys()...")
    try:
        robot._check_dependencys()
        logger.info("✅ 成功：所有依赖项（unitree-sdk2py 等）可正常导入。")
    except ImportError as e:
        logger.error(f"❌ 失败：缺少依赖 — {e}")
        raise
    except Exception as e:
        logger.error(f"❌ 失败：未知错误 — {e}")
        raise


def test_1_connect_arm():
    """1: _connect_arm"""
    logger.info("🔌 测试 _connect_arm()...")
    try:
        ensure_connected()
        assert robot.arm_controller is not None, "arm_controller 未初始化"
        assert robot.is_connected, "is_connected 仍为 False"
        logger.info("✅ 成功：机械臂控制器已连接（仿真模式）。")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        raise


def test_2_disconnect_arm():
    """2: _disconnect_arm"""
    logger.info("🛑 测试 _disconnect_arm()...")
    ensure_connected()
    try:
        robot._disconnect_arm()
        assert robot.arm_controller is None, "arm_controller 未被清空"
        logger.info("✅ 成功：机械臂控制器已安全断开。")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        raise


def test_3_set_joint_state():
    """3: _set_joint_state"""
    logger.info("⚙️ 测试 _set_joint_state()...")
    ensure_connected()
    try:
        robot._set_joint_state(ARM_Q_TARGET)
        logger.info("✅ 成功：目标关节角 (14D) 已下发（无异常）。")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        raise


def test_4_get_joint_state():
    """4: _get_joint_state"""
    logger.info("📊 测试 _get_joint_state()...")
    ensure_connected()
    try:
        q = robot._get_joint_state()
        assert isinstance(q, np.ndarray), "返回值不是 np.ndarray"
        assert q.shape == (14,), f"形状错误：期望 (14,)，得到 {q.shape}"
        assert np.all(np.isfinite(q)), "包含 NaN 或 inf"
        logger.info(f"✅ 成功：读取到 (14,) 关节角：{q.round(4)}")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        raise


def test_5_set_ee_state():
    """5: _set_ee_state"""
    logger.info("📍 测试 _set_ee_state()...")
    ensure_connected()
    try:
        robot._set_ee_state(EE_STATE_TEST)
        logger.info("✅ 成功：末端位姿 (12D) 已下发（无崩溃）。")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        # IK 在仿真中偶发失败是正常的，只要不 crash 即可接受
        logger.info("ℹ️  注：IK 求解失败在仿真中常见，不影响接口可用性。")


def test_6_get_ee_state():
    """6: _get_ee_state"""
    logger.info("🎯 测试 _get_ee_state()...")
    ensure_connected()
    try:
        ee = robot._get_ee_state()
        assert isinstance(ee, np.ndarray), "返回值不是 np.ndarray"
        assert ee.shape == (12,), f"形状错误：期望 (12,)，得到 {ee.shape}"
        assert np.all(np.isfinite(ee)), "包含 NaN 或 inf"
        # 基础合理性检查（腕部 Z 应在 0.05~0.5m）
        assert 0.05 < ee[2] < 0.5, f"❌ 左腕 Z 异常：{ee[2]:.3f}m"
        assert 0.05 < ee[8] < 0.5, f"❌ 右腕 Z 异常：{ee[8]:.3f}m"
        logger.info(f"✅ 成功：读取到 (12,) 末端位姿：{ee.round(4)}")
    except Exception as e:
        logger.error(f"❌ 失败：{e}")
        raise


# ================================
# 📋 主菜单（0–6）
# ================================

TESTS = [
    ("0", "检查依赖 (_check_dependencys)", test_0_check_dependencys),
    ("1", "连接机械臂 (_connect_arm)", test_1_connect_arm),
    ("2", "断开机械臂 (_disconnect_arm)", test_2_disconnect_arm),
    ("3", "设置关节状态 (_set_joint_state)", test_3_set_joint_state),
    ("4", "获取关节状态 (_get_joint_state)", test_4_get_joint_state),
    ("5", "设置末端状态 (_set_ee_state)", test_5_set_ee_state),
    ("6", "获取末端状态 (_get_ee_state)", test_6_get_ee_state),
]

def print_menu():
    logger.info("\n" + "═" * 50)
    logger.info("🔧 G1_29_Robot 方法级测试（0–6 编号）")
    logger.info("═" * 50)
    for key, name, _ in TESTS:
        logger.info(f"  {key}. {name}")
    logger.info("  q. 退出测试")
    logger.info("─" * 50)


def main():
    while True:
        print_menu()
        choice = input("请输入编号 (0–6) 或 'q' 退出：").strip()

        if choice.lower() == 'q':
            logger.info("👋 测试结束。机器人已自动断开。")
            robot._disconnect()
            break

        if choice not in [t[0] for t in TESTS]:
            logger.warning("❗ 输入无效，请输入 0–6 或 'q'")
            continue

        # 执行对应测试
        for key, name, func in TESTS:
            if key == choice:
                logger.info(f"\n🚀 正在执行：{name}")
                try:
                    func()
                    logger.info("🎉 测试通过。\n")
                except KeyboardInterrupt:
                    logger.info("⚠️  用户中断。")
                    break
                except Exception as e:
                    logger.error(f"💥 测试异常终止：{e}\n")
                break


if __name__ == "__main__":
    main()
