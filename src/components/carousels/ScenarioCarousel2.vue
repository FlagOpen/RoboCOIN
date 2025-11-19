<template>
  <div class="scenario-showcase">
    <!-- 左侧分类栏 -->
    <div class="category-bar">
      <div 
        v-for="(category, index) in categoryList" 
        :key="index"
        class="category-item"
        :class="{ active: currentCategoryIndex === index }"
        @click="switchCategory(index)"
        :style="getCategoryItemStyle(category)"
      >
        <span class="category-name">{{ formatCategoryName(category) }}</span>
      </div>
    </div>

    <!-- 右侧场景展示区 -->
    <div class="scenario-display">
      <transition :name="transitionName" mode="out-in">
        <div class="scenario-slide" :key="currentScenario.id">
          <img :src="currentScenario.image" :alt="currentScenario.name" class="scenario-image" />
          <div class="scenario-info">
            <h3 class="scenario-name">{{ currentScenario.name }}</h3>
            <div v-if="currentScenario.features" class="scenario-features">
              <span class="feature-tag">{{ currentScenario.features }}</span>
            </div>
          </div>
        </div>
      </transition>
      
      <!-- 指示器 -->
      <div class="indicators">
        <span 
          v-for="(scenario, index) in currentScenarios" 
          :key="scenario.id"
          class="indicator"
          :class="{ active: currentScenarioIndex === index }"
          @click="goToScenario(index)"
        ></span>
      </div>
    </div>
  </div>
</template>

<script>
const getImageUrl = (imageName) => {
  return new URL(`../../assets/scenarios/${imageName}`, import.meta.url).href
}

export default {
  name: 'ScenarioShowcase',
  data() {
    return {
      currentCategoryIndex: 0,
      currentScenarioIndex: 0,
      timer: null,
      transitionName: 'slide-left',
      scenariosData: {
        residential: [
          { 
            id: 1, 
            name: 'Home', 
            image: getImageUrl('home.png'), 
            features: '' 
          },
          { 
            id: 2, 
            name: 'Living Room', 
            image: getImageUrl('living_room.png'), 
            features: '' 
          }
        ],
        commercial: [
          { 
            id: 1, 
            name: 'Kitchen', 
            image: getImageUrl('kitchen.png'), 
            features: '' 
          },
          { 
            id: 2, 
            name: 'Market', 
            image: getImageUrl('market.png'), 
            features: '' 
          }
        ],
        working: [
          { 
            id: 1, 
            name: 'Factory', 
            image: getImageUrl('factory.png'), 
            features: '' 
          },
          { 
            id: 2, 
            name: 'School', 
            image: getImageUrl('school.png'), 
            features: '' 
          }
        ]
      }
    }
  },
  computed: {
    categoryList() {
      return Object.keys(this.scenariosData)
    },
    currentScenarios() {
      const categoryKey = this.categoryList[this.currentCategoryIndex]
      return this.scenariosData[categoryKey] || []
    },
    currentScenario() {
      return this.currentScenarios[this.currentScenarioIndex] || {}
    }
  },
  methods: {
    getImageUrl(path) {
      return getImageUrl(path);
    },
    
    getCategoryItemStyle(category) {
      const categoryScenarios = this.scenariosData[category];
      if (categoryScenarios && categoryScenarios.length > 0) {
        const bgImage = categoryScenarios[0].image;
        return {
          backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)), url(${bgImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        };
      }
      return {};
    },
    
    nextScenario() {
      this.transitionName = 'slide-left'
      
      if (this.currentScenarioIndex < this.currentScenarios.length - 1) {
        this.currentScenarioIndex++
      } else {
        this.currentScenarioIndex = 0
        this.currentCategoryIndex = (this.currentCategoryIndex + 1) % this.categoryList.length
      }
    },
    
    prevScenario() {
      this.transitionName = 'slide-right'
      
      if (this.currentScenarioIndex > 0) {
        this.currentScenarioIndex--
      } else {
        this.currentCategoryIndex = (this.currentCategoryIndex - 1 + this.categoryList.length) % this.categoryList.length
        this.currentScenarioIndex = this.currentScenarios.length - 1
      }
    },
    
    switchCategory(index) {
      this.transitionName = index > this.currentCategoryIndex ? 'slide-left' : 'slide-right'
      this.currentCategoryIndex = index
      this.currentScenarioIndex = 0
      this.resetTimer()
    },
    
    goToScenario(index) {
      this.transitionName = index > this.currentScenarioIndex ? 'slide-left' : 'slide-right'
      this.currentScenarioIndex = index
      this.resetTimer()
    },
    
    startAutoPlay() {
      this.timer = setInterval(() => {
        this.nextScenario()
      }, 3000)
    },
    
    resetTimer() {
      if (this.timer) {
        clearInterval(this.timer)
      }
      this.startAutoPlay()
    },
    
    formatCategoryName(category) {
      const nameMap = {
        residential: 'Residential',
        commercial: 'Commercial',
        working: 'Working'
      }
      return nameMap[category] || category
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
.scenario-showcase {
  display: flex;
  /* font-family: 'Arial', sans-serif; */
  height: 400px;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.category-bar {
  width: 40%;
  background-color: #f8f9fa;
  display: flex;
  flex-direction: column;
  position: relative;
  z-index: 1;
}

.category-item {
  cursor: pointer;
  transition: all 0.3s ease;
  border-left: 4px solid transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-grow: 1;
  position: relative;
}

.category-item:hover {
  background-color: rgba(0, 0, 0, 0.03);
  transform: translateX(5px);
}

.category-item.active {
  border-left-color: #4a90e2;
  background-color: rgba(74, 144, 226, 0.08);
  font-weight: bold;
}

.category-name {
  writing-mode: horizontal-tb;
  font-size: 1.3rem;
  letter-spacing: 1px;
  font-weight: 600;
  text-align: center;
  color: #2c3e50;
  z-index: 2;
}

.scenario-display {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  position: relative;
  overflow: hidden;
}

.scenario-slide {
  text-align: center;
  max-width: 85%;
}

.scenario-image {
  width: 100%;
  object-fit: contain;
  border-radius: 8px;
  transition: transform 0.3s ease;
}

.scenario-name {
  font-size: 1rem;
  margin-top: 20px;
}

.feature-tag {
  display: inline-block;
  background: linear-gradient(45deg, #ff6b6b, #ee5a24);
  color: white;
  border-radius: 20px;
  font-size: 1rem;
  font-weight: 500;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.indicators {
  position: absolute;
  bottom: 20px;
  display: flex;
  gap: 12px;
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
  transition: all 0.6s ease;
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
  .scenario-showcase {
    flex-direction: column;
    height: auto;
  }
  
  .category-bar {
    width: 100%;
    flex-direction: row;
    height: auto;
  }
  
  .category-item {
    border-left: none;
    border-bottom: 4px solid transparent;
    flex-grow: 1;
    justify-content: center;
    min-height: 80px;
  }
  
  .category-item.active {
    border-left-color: transparent;
    border-bottom-color: #4a90e2;
  }
  
  .category-name {
    font-size: 1.1rem;
  }
  
  .scenario-image {
    max-height: 40vh;
    height: auto;
  }
  
  .scenario-name {
    font-size: 1.5rem;
  }
}

@media (max-width: 480px) {
  .category-name {
    font-size: 1rem;
  }
  
  .scenario-name {
    font-size: 1.3rem;
  }
  
  .feature-tag {
    font-size: 0.9rem;
  }
}
</style>