<template>
  <div class="robot-showcase">
    <h2>Robotic Platforms</h2>
    <!-- 第一行：类型选择按钮 -->
    <div class="type-buttons">
      <button 
        v-for="type in robotTypes" 
        :key="type.id"
        @click="selectType(type.id)"
        :class="{ active: currentType === type.id }"
        class="type-button"
      >
        <img :src="type.icon" :alt="type.name" class="type-icon" />
        <span class="type-name">{{ type.name }}</span>
      </button>
    </div>
    
    <!-- 第二行：机器人展示区域 -->
    <div class="robot-display">
      <transition-group name="fade" tag="div" class="robot-grid">
        <div 
          v-for="robot in currentRobots" 
          :key="robot.id"
          class="robot-card"
        >
          <div class="robot-image-container">
            <img :src="robot.image" :alt="robot.name" class="robot-image" />
          </div>
          <div class="robot-info">
            <h3 class="robot-title">{{ robot.name }}</h3>
            <p v-if="robot.features" class="robot-features">{{ robot.features }}</p>
          </div>
        </div>
      </transition-group>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

// 动态导入图片函数
const getImageUrl = (imageName) => {
  return new URL(`../assets/robots/${imageName}`, import.meta.url).href
}

// 机器人类型数据
const robotTypes = ref([
  {
    id: 'dual',
    name: 'Dual',
    icon: getImageUrl('dual.png')
  },
  {
    id: 'half-humanoid',
    name: 'Half-Humanoid',
    icon: getImageUrl('half-humanoid.png')
  },
  {
    id: 'humanoid',
    name: 'Humanoid',
    icon: getImageUrl('humanoid.png')
  }
])

// 所有机器人数据
const allRobots = ref({
  dual: [
    {
      id: 1,
      name: 'Agilex COBOT MAGIC',
      image: getImageUrl('agilex cobot magic.png'),
      features: ''
    },
    {
      id: 2,
      name: 'Galaxea R1 Lite',
      image: getImageUrl('galaxea r1 lite.png'),
      features: ''
    },
    {
      id: 3,
      name: 'FutureWei ADora',
      image: getImageUrl('galaxea r1 lite.png'),
      features: ''
    }
  ],
  'half-humanoid': [
    {
      id: 1,
      name: 'Galbot G1',
      image: getImageUrl('galbot g1.png'),
      features: ''
    },
    {
      id: 2,
      name: 'Tianqing A2',
      image: getImageUrl('tianqing a2.png'),
      features: ''
    },
    {
      id: 3,
      name: 'Realman AIDA-L',
      image: getImageUrl('realman aida l.png'),
      features: ''
    },
    {
      id: 4,
      name: 'AI2 Alphabot 2',
      image: getImageUrl('ai2 alphabot 2.png'),
      features: 'Dexterous Hand'
    },
    {
      id: 5,
      name: 'AIRBOT MMK2',
      image: getImageUrl('airbot mmk2.png'),
      features: 'Dexterous Hand'
    }
  ],
  humanoid: [
    {
      id: 1,
      name: 'LEJU Kuavo 4 Pro',
      image: getImageUrl('leju kuavo 4 pro.png'),
      features: 'Dexterous Hand'
    },
    {
      id: 2,
      name: 'Unitree G1edu',
      image: getImageUrl('unitree g1edu.png'),
      features: 'Dexterous Hand'
    }
  ]
})

// 响应式数据
const currentType = ref('dual')
let autoSwitchTimer = null
const switchInterval = 3000 // 3秒切换间隔

// 计算当前显示的机器人
const currentRobots = computed(() => {
  return allRobots.value[currentType.value] || []
})

// 选择类型
const selectType = (typeId) => {
  currentType.value = typeId
  // 重置自动切换计时器
  restartAutoSwitch()
}

// 切换到下一个类型
const switchToNextType = () => {
  const currentIndex = robotTypes.value.findIndex(type => type.id === currentType.value)
  const nextIndex = (currentIndex + 1) % robotTypes.value.length
  currentType.value = robotTypes.value[nextIndex].id
}

// 开始自动切换
const startAutoSwitch = () => {
  stopAutoSwitch() // 确保没有重复的定时器
  autoSwitchTimer = setInterval(switchToNextType, switchInterval)
}

// 停止自动切换
const stopAutoSwitch = () => {
  if (autoSwitchTimer) {
    clearInterval(autoSwitchTimer)
    autoSwitchTimer = null
  }
}

// 重新启动自动切换
const restartAutoSwitch = () => {
  stopAutoSwitch()
  startAutoSwitch()
}

// 生命周期钩子
onMounted(() => {
  // 组件挂载后启动自动切换
  startAutoSwitch()
})

onUnmounted(() => {
  // 组件销毁时清理定时器
  stopAutoSwitch()
})
</script>

<style scoped>
h2 {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 1.5rem;
  color: #2c3e50;
}

.robot-showcase {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  background: transparent;
  min-height: 100vh;
}

.type-buttons {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
}

.type-button {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1.5rem 1rem;
  border: 1px solid #fff;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: none;
}

.type-button:hover {
  transform: translateY(-2px);
  box-shadow: none;
  border-color: #3498db;
}

.type-button.active {
  border-color: #3498db;
  background-color: transparent;
  box-shadow: none;
}

.type-icon {
  height: 250px;
  object-fit: contain;
  margin-bottom: 0.8rem;
  filter: drop-shadow(0 1px 1px rgba(0,0,0,0.1));
}

.type-name {
  font-size: 1.1rem;
  font-weight: 600;
  color: #34495e;
}

.robot-display {
  display: flex;
  justify-content: center;
  overflow-x: auto; /* 添加水平滚动 */
  padding: 1rem 0;
}

.robot-grid {
  display: flex; /* 改为flex布局实现不换行 */
  flex-wrap: nowrap; /* 禁止换行 */
  gap: 2rem; /* 卡片间距 */
  padding: 1rem;
  justify-content: flex-start; /* 左对齐 */
  min-width: min-content; /* 确保容器足够宽以容纳所有卡片 */
}

.robot-card {
  border-radius: 8px;
  overflow: hidden;
  background: transparent;
  box-shadow: none;
  transition: all 0.3s ease;
  min-width: 50px;
  /* height: 250px; */
  flex-shrink: 0; /* 防止卡片被压缩 */
}


.robot-card:hover {
  transform: translateY(-2px);
  box-shadow: none;
}

.robot-image-container {
  height: auto;
  overflow: hidden;
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
}

.robot-image {
  /* width: 50%; */
  /* aspect-ratio: 1/2; */
  height: 200px;
  object-fit: contain;
  transition: transform 0.3s ease;
}

.robot-card:hover .robot-image {
  transform: scale(1.02);
}

.robot-info {
  padding: 1.5rem;
  text-align: center;
  background: transparent;
}

.robot-title {
  margin: 0 0 0.5rem 0;
  color: #2c3e50;
  font-size: 1.2rem;
  font-weight: 600;
  line-height: 1.4;
}

.robot-features {
  margin: 0;
  color: #7f8c8d;
  font-size: 0.95rem;
  font-style: italic;
}

/* 这是修复布局跳跃问题的核心CSS */
.robot-grid {
  position: relative; /* 为绝对定位的离开元素建立容器 */
}

.fade-move {
  transition: transform 0.5s ease; /* 移动动画 */
}

.fade-enter-active,
.fade-leave-active {
  transition: all 0.5s ease; /* 进入和离开的动画 */
}

.fade-enter-from {
  opacity: 0;
  transform: translateY(20px); /* 元素从下方淡入 */
}

.fade-leave-to {
  opacity: 0;
  transform: scale(0.9); /* 元素缩小淡出 */
}

.fade-leave-active {
  position: absolute; /* 关键：使离开元素不占位 */
  width: 100%; /* 可选的，帮助保持布局稳定 */
}

/* 移除move动画相关样式 */

/* 响应式设计 */
@media (max-width: 768px) {
  .robot-showcase {
    padding: 1rem;
  }
  
  .type-buttons {
    flex-direction: column;
    align-items: center;
    gap: 1rem;
  }
  
  .type-button {
    width: 30%;
    max-width: 250px;
    flex-direction: row;
    justify-content: flex-start;
    gap: 1rem;
  }
  
  .type-icon {
    width: 25%;
    margin-bottom: 0;
  }
  
  .robot-grid {
    gap: 1rem; /* 移动端减小间距 */
  }
  
  .robot-card {
    min-width: 180px; /* 移动端稍小宽度 */
  }
}

@media (max-width: 480px) {
  .robot-showcase {
    padding: 0.5rem;
  }
  
  .type-name {
    font-size: 1rem;
  }
  
  .robot-title {
    font-size: 1.1rem;
  }
  
  .robot-card {
    min-width: 160px; /* 小屏幕更小宽度 */
  }
}
</style>