<template>
  <div class="robot-showcase">
    <!-- 左侧标题栏 - 整体竖排 -->
    <div class="category-bar">
      <div 
        v-for="(category, index) in categoryList" 
        :key="index"
        class="category-item"
        :class="{ active: currentCategoryIndex === index }"
        @click="switchCategory(index)"
        :style="getCategoryItemStyle(category)"
      >
        <!-- 修改：文字横排 -->
        <span class="category-name">{{ formatCategoryName(category) }}</span>
      </div>
    </div>

    <!-- 右侧机器人展示区 -->
    <div class="robot-display">
      <transition :name="transitionName" mode="out-in">
        <div class="robot-slide" :key="currentRobot.id">
          <img :src="currentRobot.image" :alt="currentRobot.name" class="robot-image" />
          <div class="robot-info">
            <h3 class="robot-name">{{ currentRobot.name }}</h3>
            <div v-if="currentRobot.features" class="robot-features">
              <span class="feature-tag">{{ currentRobot.features }}</span>
            </div>
          </div>
        </div>
      </transition>
      
      <!-- 指示器 -->
      <div class="indicators">
        <span 
          v-for="(robot, index) in currentRobots" 
          :key="robot.id"
          class="indicator"
          :class="{ active: currentRobotIndex === index }"
          @click="goToRobot(index)"
        ></span>
      </div>
    </div>
  </div>
</template>

<script>
const getImageUrl = (imageName) => {
  return new URL(`../../assets/robots/${imageName}`, import.meta.url).href
}

export default {
  name: 'RobotShowcase',
  data() {
    return {
      currentCategoryIndex: 0,
      currentRobotIndex: 0,
      timer: null,
      transitionName: 'slide-left',
      robotsData: {
        dual: [
          { id: 1, name: 'Agilex COBOT MAGIC', image: getImageUrl('agilex cobot magic.png'), features: '' },
          { id: 2, name: 'Galaxea R1 Lite', image: getImageUrl('galaxea r1 lite.png'), features: '' },
          { id: 3, name: 'FutureWei ADora', image: getImageUrl('galaxea r1 lite.png'), features: '' }
        ],
        'half-humanoid': [
          { id: 1, name: 'Galbot G1', image: getImageUrl('galbot g1.png'), features: '' },
          { id: 2, name: 'Tianqing A2', image: getImageUrl('tianqing a2.png'), features: '' },
          { id: 3, name: 'Realman AIDA-L', image: getImageUrl('realman aida l.png'), features: '' },
          { id: 4, name: 'AI2 Alphabot 2', image: getImageUrl('ai2 alphabot 2.png'), features: '' },
          { id: 5, name: 'AIRBOT MMK2', image: getImageUrl('airbot mmk2.png'), features: '' }
        ],
        humanoid: [
          { id: 1, name: 'LEJU Kuavo 4 Pro', image: getImageUrl('leju kuavo 4 pro.png'), features: '' },
          { id: 2, name: 'Unitree G1edu', image: getImageUrl('unitree g1edu.png'), features: '' }
        ]
      }
    }
  },
  computed: {
    categoryList() {
      return Object.keys(this.robotsData)
    },
    currentRobots() {
      const categoryKey = this.categoryList[this.currentCategoryIndex]
      return this.robotsData[categoryKey] || []
    },
    currentRobot() {
      return this.currentRobots[this.currentRobotIndex] || {}
    }
  },
  methods: {
    getImageUrl(path) {
      return getImageUrl(path);
    },
    
    getCategoryItemStyle(category) {
      const categoryRobots = this.robotsData[category];
      if (categoryRobots && categoryRobots.length > 0) {
        const bgImage = categoryRobots[0].image;
        console.log('背景图片URL:', bgImage);
        return {
          backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url(${bgImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        };
      }
      return {};
    },
    
    nextRobot() {
      this.transitionName = 'slide-left'
      
      if (this.currentRobotIndex < this.currentRobots.length - 1) {
        this.currentRobotIndex++
      } else {
        this.currentRobotIndex = 0
        this.currentCategoryIndex = (this.currentCategoryIndex + 1) % this.categoryList.length
      }
    },
    
    prevRobot() {
      this.transitionName = 'slide-right'
      
      if (this.currentRobotIndex > 0) {
        this.currentRobotIndex--
      } else {
        this.currentCategoryIndex = (this.currentCategoryIndex - 1 + this.categoryList.length) % this.categoryList.length
        this.currentRobotIndex = this.currentRobots.length - 1
      }
    },
    
    switchCategory(index) {
      this.transitionName = index > this.currentCategoryIndex ? 'slide-left' : 'slide-right'
      this.currentCategoryIndex = index
      this.currentRobotIndex = 0
      this.resetTimer()
    },
    
    goToRobot(index) {
      this.transitionName = index > this.currentRobotIndex ? 'slide-left' : 'slide-right'
      this.currentRobotIndex = index
      this.resetTimer()
    },
    
    startAutoPlay() {
      this.timer = setInterval(() => {
        this.nextRobot()
      }, 3000)
    },
    
    resetTimer() {
      if (this.timer) {
        clearInterval(this.timer)
      }
      this.startAutoPlay()
    },
    
    formatCategoryName(category) {
      return category.split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
    }
  },
  
  mounted() {
    this.startAutoPlay()
  },
  
  beforeUnmount() {
    if (this.timer) {
      clearInterval(this.timer)
    }
  }
}
</script>

<style scoped>
/* 修改1: 整体布局改为左右结构 */
.robot-showcase {
  display: flex;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
  border-radius: 10px;
  height: 400px;
  /* font-family: 'Arial', sans-serif; */
}

/* 修改2: 左侧标题栏竖排 */
.category-bar {
  /* width: 300px; */
  width: 40%;
  background-color: #f5f5f5;
  display: flex;
  flex-direction: column; /* 关键修改：垂直排列 */
  /* padding: 20px 0; */
  /* box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1); */
  position: relative;
  z-index: 1;
}

.category-item {
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  border-left: 4px solid transparent;
  display: flex;
  align-items: center;
  justify-content: center; /* 水平居中 */
  flex-grow: 1;
  /* height: 100px; */
}

.category-item:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.category-item.active {
  border-left-color: #4a90e2;
  background-color: rgba(74, 144, 226, 0.1);
  font-weight: bold;
}

/* 修改3: 标题文字保持横排 */
.category-name {
  writing-mode: horizontal-tb; /* 关键修改：文字横排 */
  text-orientation: mixed;
  font-size: 1.2rem;
  letter-spacing: 1px;
  font-weight: 600;
  text-align: center;
}

.robot-display {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
  /* height: 300px; */
  /* width: 40%; */
}

.robot-slide {
  text-align: center;
  max-width: 80%;
}

.robot-image {
  height: 250px;
  object-fit: contain;
  border-radius: 10px;
}

.robot-info {
  margin-top: 20px;
}

.robot-name {
  font-size: 1rem;
  margin-bottom: 40px;
  color: #333;
}

.feature-tag {
  display: inline-block;
  background-color: #4a90e2;
  color: white;
  padding: 5px 10px;
  border-radius: 15px;
  font-size: 0.9rem;
}

.indicators {
  position: absolute;
  bottom: 20px;
  display: flex;
  gap: 10px;
}

.indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background-color: #ccc;
  cursor: pointer;
  transition: background-color 0.3s;
}

.indicator.active {
  background-color: #4a90e2;
}

/* 过渡动画 */
.slide-left-enter-active,
.slide-left-leave-active,
.slide-right-enter-active,
.slide-right-leave-active {
  transition: all 0.5s ease;
}

.slide-left-enter-from {
  transform: translateX(100%);
  opacity: 0;
}

.slide-left-leave-to {
  transform: translateX(-100%);
  opacity: 0;
}

.slide-right-enter-from {
  transform: translateX(-100%);
  opacity: 0;
}

.slide-right-leave-to {
  transform: translateX(100%);
  opacity: 0;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .robot-showcase {
    flex-direction: column;
    height: auto;
  }
  
  .category-bar {
    width: 100%;
    flex-direction: row; /* 移动端改为横向排列 */
    height: auto;
    padding: 10px 0;
  }
  
  .category-item {
    border-left: none;
    border-bottom: 4px solid transparent;
    flex-grow: 1;
    justify-content: center;
    padding: 15px 10px;
    min-height: 60px;
  }
  
  .category-item.active {
    border-left-color: transparent;
    border-bottom-color: #4a90e2;
  }
  
  .category-name {
    writing-mode: horizontal-tb; /* 移动端保持横排 */
    font-size: 1rem;
  }
  
  .robot-image {
    max-height: 40vh;
  }
}
</style>