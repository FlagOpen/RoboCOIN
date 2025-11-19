<template>
  <section class="robot-carousel">
    <div class="container">
      <h2 class="section-title">{{ t('highlights.scenario_swiper.title') }}</h2>

      <!-- Swiper轮播图容器 -->
      <div class="carousel-container">
        <swiper
          :modules="modules"
          :slides-per-view="4"
          :space-between="30"
          :loop="true"
          :autoplay="{
            delay: 1500,
            disableOnInteraction: false,
          }"
          :pagination="{
            clickable: true,
            dynamicBullets: true
          }"
          :navigation="{
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev'
          }"
          class="robot-swiper"
        >
          <swiper-slide v-for="(robot, index) in robots" :key="index" class="robot-slide">
            <div class="robot-card">
              <div class="robot-image">
                <img :src="robot.image" :alt="robot.name" />
                <!-- 图片遮罩层，用于添加渐变效果 -->
                <div class="image-overlay"></div>
              </div>
              <div class="robot-info">
                <h3 class="robot-name">{{ robot.name }}</h3>
                <p class="robot-type">{{ robot.type }}</p>
                <p class="robot-features" v-if="robot.features">{{ robot.features }}</p>
              </div>
            </div>
          </swiper-slide>
          
          <!-- 导航按钮 -->
          <div class="swiper-button-prev"></div>
          <div class="swiper-button-next"></div>
          
          <!-- 分页器 -->
          <div class="swiper-pagination"></div>
        </swiper>
      </div>
    </div>
  </section>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
// 引入Swiper组件和功能模块
import { Swiper, SwiperSlide } from 'swiper/vue';
import { Autoplay, Pagination, Navigation } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';

const { t } = useI18n();

// 注册Swiper模块
const modules = [Autoplay, Pagination, Navigation];

// 机器人数据
const robots = [
  {
    image: new URL('../../assets/scenarios/bedroom.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.bed'),
    type: t('highlights.scenario_swiper.residential'),
  },
  {
    // cloth
    image: new URL('../../assets/scenarios/cloth.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.cloth'),
    type: t('highlights.scenario_swiper.residential'),
  },
  {
    // food
    image: new URL('../../assets/scenarios/food.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.food'),
    type: t('highlights.scenario_swiper.residential'),
  },
  {
    // kitchen
    image: new URL('../../assets/scenarios/kitchen.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.kitchen'),
    type: t('highlights.scenario_swiper.residential'),
  },
  {
    // toilet
    image: new URL('../../assets/scenarios/toilet.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.toilet'),
    type: t('highlights.scenario_swiper.residential'),
  },
  {
    // supermarket
    image: new URL('../../assets/scenarios/supermarket.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.supermarket'),
    type: t('highlights.scenario_swiper.commercial'),
  },
  {
    // icecream
    image: new URL('../../assets/scenarios/icecream.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.icecream'),
    type: t('highlights.scenario_swiper.commercial'),
  },
  {
    // pipeline
    image: new URL('../../assets/scenarios/pipeline.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.pipeline'),
    type: t('highlights.scenario_swiper.working'),
  },
  {
    // recycle
    image: new URL('../../assets/scenarios/recycle.png', import.meta.url).href,
    name: t('highlights.scenario_swiper.recycle'),
    type: t('highlights.scenario_swiper.working'),
  }
];

// 获取图片URL
const getImageUrl = (imageName) => {
  return `/robots/${imageName}`;
};
</script>

<style scoped>
.robot-carousel {
  margin-top: 2rem;
  padding: 1rem 1rem;
  border: 1px solid #e1e5e9;
  border-radius: 12px;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  position: relative;
}

.section-title {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 2rem;
  color: #2c3e50;
  font-weight: 700;
}

.carousel-container {
  height: auto;
  border-radius: 16px;
  overflow: hidden;
  background: white;
  padding: 1rem 0;
  position: relative;
}

.robot-swiper {
  width: 100%;
  height: 100%;
  position: relative;
  padding: 0 40px; /* 为导航按钮留出空间 */
}

.robot-slide {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem 0.5rem;
}

.robot-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  height: 100%;
  border-radius: 16px;
  overflow: hidden;
  transition: all 0.3s ease;
  background: white;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.robot-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.robot-image {
  position: relative;
  width: 100%;
  height: 0;
  padding-bottom: 75%; /* 4:3 固定比例 */
  border-radius: 12px 12px 0 0;
  overflow: hidden;
  background: #f8fafc;
}

.robot-image img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.5s ease;
  border-radius: 12px 12px 0 0;
}

.robot-card:hover .robot-image img {
  transform: scale(1.05);
}

.image-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    transparent 70%,
    rgba(0, 0, 0, 0.1) 100%
  );
  border-radius: 12px 12px 0 0;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.robot-card:hover .image-overlay {
  opacity: 1;
}

.robot-info {
  width: 100%;
  padding: 1rem 1rem;
  text-align: center;
  background: white;
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.robot-name {
  font-size: 1.1rem;
  font-weight: 600;
  line-height: 1.3;
  color: #2c3e50;
}

.robot-type {
  font-size: 1.1rem;
  color: #3498db;
  font-weight: 500;
  letter-spacing: 0.5px;
}

.robot-features {
  font-size: 0.85rem;
  color: #7f8c8d;
  font-style: italic;
  margin: 0.5rem 0 0 0;
  line-height: 1.4;
}

/* Swiper导航按钮样式 - 修改后的样式 */
.swiper-button-prev,
.swiper-button-next {
  color: #3498db;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  transition: all 0.3s ease;
  /* 移除白色背景和阴影 */
  background: transparent !important;
  box-shadow: none !important;
  margin-top: -20px; /* 垂直居中 */
}

.swiper-button-prev {
  left: 0; /* 与容器左边缘对齐 */
}

.swiper-button-next {
  right: 0; /* 与容器右边缘对齐 */
}

.swiper-button-prev:hover,
.swiper-button-next:hover {
  color: #2980b9;
  background: rgba(52, 152, 219, 0.1) !important; /* 悬停时添加半透明背景 */
  transform: scale(1.1);
}

.swiper-button-prev::after,
.swiper-button-next::after {
  font-size: 1.5rem; /* 增大箭头大小 */
  font-weight: bold;
}

/* Swiper分页器样式 */
.swiper-pagination {
  bottom: 10px !important;
}

.swiper-pagination-bullet {
  width: 10px;
  height: 10px;
  opacity: 0.7;
  background: #bdc3c7;
  transition: all 0.3s ease;
}

.swiper-pagination-bullet-active {
  background: #3498db;
  opacity: 1;
  transform: scale(1.2);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .robot-swiper {
    padding: 0 35px; /* 调整内边距 */
  }
  
  .robot-slide {
    padding: 1rem 0.25rem;
  }
  
  .robot-info {
    padding: 1rem 0.5rem;
  }
  
  .robot-name {
    font-size: 1rem;
  }
  
  .robot-type {
    font-size: 0.85rem;
  }
  
  .swiper-button-prev,
  .swiper-button-next {
    width: 35px;
    height: 35px;
    margin-top: -17.5px;
  }
  
  .swiper-button-prev::after,
  .swiper-button-next::after {
    font-size: 1.3rem;
  }
}

@media (max-width: 1024px) {
  .carousel-container {
    min-height: 380px;
  }
  
  .robot-image {
    padding-bottom: 70%; /* 调整为更宽的比例 */
  }
  
  .robot-swiper {
    padding: 0 30px;
  }
}

@media (max-width: 768px) {
  .robot-carousel {
    padding: 1.5rem 0.5rem;
  }
  
  .section-title {
    font-size: 1.75rem;
    margin-bottom: 1.5rem;
  }
  
  .carousel-container {
    min-height: 350px;
    padding: 0.5rem 0;
  }
  
  .robot-swiper {
    padding: 0 25px;
  }
  
  .robot-slide {
    padding: 0.5rem;
  }
  
  .robot-card {
    border-radius: 12px;
  }
  
  .robot-image {
    border-radius: 10px 10px 0 0;
  }
  
  .robot-image img {
    border-radius: 10px 10px 0 0;
  }
  
  .image-overlay {
    border-radius: 10px 10px 0 0;
  }
  
  .robot-info {
    padding: 1rem 0.5rem;
  }
  
  .robot-name {
    font-size: 1rem;
  }
  
  .robot-type {
    font-size: 0.8rem;
  }
  
  .swiper-button-prev,
  .swiper-button-next {
    width: 30px;
    height: 30px;
    margin-top: -15px;
  }
  
  .swiper-button-prev::after,
  .swiper-button-next::after {
    font-size: 1.1rem;
  }
}

@media (max-width: 640px) {
  .robot-carousel {
    padding: 1rem 0.25rem;
  }
  
  .section-title {
    font-size: 1.5rem;
  }
  
  .carousel-container {
    min-height: 320px;
  }
  
  .robot-swiper {
    padding: 0 20px;
  }
  
  .robot-image {
    padding-bottom: 65%; /* 移动端使用更宽的比例 */
  }
  
  .robot-info {
    padding: 0.75rem 0.25rem;
  }
  
  .robot-name {
    font-size: 0.9rem;
  }
  
  .robot-type {
    font-size: 0.75rem;
  }
  
  .swiper-button-prev,
  .swiper-button-next {
    width: 25px;
    height: 25px;
    margin-top: -12.5px;
  }
  
  .swiper-button-prev::after,
  .swiper-button-next::after {
    font-size: 0.9rem;
  }
}

@media (max-width: 480px) {
  .container {
    padding: 0 0.5rem;
  }
  
  .carousel-container {
    min-height: 300px;
  }
  
  .robot-swiper {
    padding: 0 15px;
  }
  
  .robot-slide {
    padding: 0.25rem;
  }
  
  .robot-image {
    padding-bottom: 60%;
  }
  
  .robot-info {
    padding: 0.5rem 0.25rem;
  }
  
  .robot-name {
    font-size: 0.85rem;
  }
  
  .robot-type {
    font-size: 0.7rem;
  }
  
  .swiper-button-prev,
  .swiper-button-next {
    display: none; /* 在小屏幕上隐藏导航按钮 */
  }
}
</style>