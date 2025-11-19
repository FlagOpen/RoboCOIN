<template>
  <section class="annotation-process">
    <div class="container">
      <h2 class="section-title">{{ t('corobot.title') }}</h2>
      <div class="animation-container">
        <p class="description">
          {{ t('corobot.desc') }}
        </p>
        <div class="carousel-container">
          <!-- 图片显示区域 -->
          <div class="image-wrapper">
            <img :src="currentImage.src" :alt="currentImage.title" class="carousel-image" />
            <!-- 右上角说明文字 -->
            <div class="image-caption">
              <h3 class="caption-title">{{ currentImage.title }}</h3>
              <p class="caption-desc">{{ currentImage.description }}</p>
            </div>
          </div>

          <!-- 指示器 -->
          <div class="carousel-indicators">
            <div
              v-for="(image, index) in images"
              :key="index"
              :class="['indicator', { active: currentIndex === index }]"
              @click="switchToImage(index)"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// 响应式数据
const currentIndex = ref(0)
const autoPlayInterval = ref(null)

// 图片数据
const images = [
  {
    src: new URL('../assets/corobot/corobot_collect.png', import.meta.url).href,
    title: 'DataCollect',
    description: t('corobot.modules.data_collect')
  },
  {
    src: new URL('../assets/corobot/corobot_convert.png', import.meta.url).href,
    title: 'DataConvert',
    description: t('corobot.modules.data_convert')
  },
  {
    src: new URL('../assets/corobot/corobot_forge.png', import.meta.url).href,
    title: 'DataForge',
    description: t('corobot.modules.data_forge')
  },
  {
    src: new URL('../assets/corobot/corobot_robocoin.png', import.meta.url).href,
    title: 'RoboCOIN',
    description: t('corobot.modules.robocoin')
  },
  {
    src: new URL('../assets/corobot/corobot_manage.png', import.meta.url).href,
    title: 'DataManage',
    description: t('corobot.modules.data_manage')
  },
  {
    src: new URL('../assets/corobot/corobot_train.png', import.meta.url).href,
    title: 'DataTrain',
    description: t('corobot.modules.data_train')
  }
]

// 计算属性
const currentImage = computed(() => images[currentIndex.value])

// 方法定义
// 切换到下一张图片
const nextImage = () => {
  currentIndex.value = (currentIndex.value + 1) % images.length
}

// 直接切换到指定图片
const switchToImage = (index) => {
  currentIndex.value = index
  // 切换图片后重置自动播放计时器
  resetAutoPlay()
}

// 开始自动播放
const startAutoPlay = () => {
  if (autoPlayInterval.value) {
    clearInterval(autoPlayInterval.value)
  }
  autoPlayInterval.value = setInterval(() => {
    nextImage()
  }, 5000) // 每3秒切换一次
}

// 停止自动播放
const stopAutoPlay = () => {
  if (autoPlayInterval.value) {
    clearInterval(autoPlayInterval.value)
    autoPlayInterval.value = null
  }
}

// 重置自动播放（切换后重新计时）
const resetAutoPlay = () => {
  stopAutoPlay()
  startAutoPlay()
}

// 生命周期钩子
// 组件挂载后开始自动播放
onMounted(() => {
  startAutoPlay()
})

// 组件卸载前清除定时器，防止内存泄漏
onUnmounted(() => {
  stopAutoPlay()
})
</script>

<style scoped>
.annotation-process {
  padding: 2rem 12%;
}

.container {
  /* max-width: 1200px; */
  margin: 0 auto;
}

.section-title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 3rem;
  color: #2c3e50;
}

.description {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  color: #2c3e50;
  line-height: 1.6;
}

.animation-container {
  width: 100%;
}

.carousel-container {
  position: relative;
  width: 100%;
  /* max-width: 900px; */
  margin: 0 auto;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  background: #fff;
}

.image-wrapper {
  position: relative;
  width: 100%;
  /* height: 500px; */
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8f9fa;
}

.carousel-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.image-caption {
  position: absolute;
  top: 20px;
  right: 20px;
  /* background: rgba(255, 255, 255, 0.95); */
  padding: 20px;
  border-radius: 8px;
  /* box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1); */
  max-width: 400px;
  /* backdrop-filter: blur(10px); */
  /* border: 1px solid rgba(255, 255, 255, 0.2); */
}

.caption-title {
  font-size: 1.5rem;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 8px 0;
}

.caption-desc {
  font-size: 1rem;
  color: #5a6c7d;
  margin: 0;
  line-height: 1.5;
}

.carousel-indicators {
  position: absolute;
  bottom: 20px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  gap: 12px;
  padding: 0 20px;
}

.indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.6);
  cursor: pointer;
  transition: all 0.3s ease;
  border: 2px solid transparent;
}

.indicator:hover {
  background: rgba(255, 255, 255, 0.9);
  transform: scale(1.2);
}

.indicator.active {
  background: #2c3e50;
  transform: scale(1.2);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .annotation-process {
    padding: 2rem 5%;
  }
  
  .section-title {
    font-size: 2rem;
  }
  
  .description {
    font-size: 1.1rem;
  }
  
  .image-wrapper {
    height: 400px;
  }
  
  .image-caption {
    position: relative;
    top: auto;
    right: auto;
    max-width: none;
    margin: 20px;
  }
}
</style>