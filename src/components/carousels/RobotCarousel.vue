<template>
  <section class="robot-carousel">
    <div class="container">
      <h2 class="section-title">{{ t('highlights.swiper.title') }}</h2>

      <!-- Swiper轮播图容器 -->
      <div class="carousel-container">
        <swiper
          :modules="modules"
          :slides-per-view="5"
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
    image: new URL('../../assets/robots/agilex split aloha.png', import.meta.url).href,
    name: 'Agilex Split Aloha',
    type: t('highlights.swiper.dual')
  },
  {
    image: new URL('../../assets/robots/agilex cobot magic.png', import.meta.url).href,
    name: 'Agilex Cobot Magic',
    type: t('highlights.swiper.dual')
  },
  {
    image: new URL('../../assets/robots/galaxea r1 lite.png', import.meta.url).href,
    name: 'Galaxea R1 Lite',
    type: t('highlights.swiper.dual')
  },
  {
    image: new URL('../../assets/robots/galbot g1.png', import.meta.url).href,
    name: 'Galbot G1',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/realman aida l.png', import.meta.url).href,
    name: 'Realman RMC-AIDA-L',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/realman rs01.png', import.meta.url).href,
    name: 'Realman RS-01',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/realman rs02.png', import.meta.url).href,
    name: 'Realman RS-02',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/tianqing a2.png', import.meta.url).href,
    name: 'Tianqing A2',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/tianqing a2d.png', import.meta.url).href,
    name: 'Tianqing A2-D',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/leju kuavo 4 lb.png', import.meta.url).href,
    name: 'LEJU Kuavo 4 LB',
    type: t('highlights.swiper.half_humanoid')
  },
  {
    image: new URL('../../assets/robots/airbot mmk2.png', import.meta.url).href,
    name: 'AIRBOT MMK2',
    type: t('highlights.swiper.half_humanoid'),
    features: t('highlights.swiper.dexterous')
  },
  {
    image: new URL('../../assets/robots/ai2 alphabot 1s.png', import.meta.url).href,
    name: 'AI2 Alphabot 1s',
    type: t('highlights.swiper.half_humanoid'),
    features: t('highlights.swiper.dexterous')
  },
  {
    image: new URL('../../assets/robots/ai2 alphabot 2.png', import.meta.url).href,
    name: 'AI2 Alphabot 2',
    type: t('highlights.swiper.half_humanoid'),
    features: t('highlights.swiper.dexterous')
  },
  {
    image: new URL('../../assets/robots/leju kuavo 4 pro.png', import.meta.url).href,
    name: 'LEJU Kuavo 4 Pro',
    type: t('highlights.swiper.humanoid'),
    features: t('highlights.swiper.dexterous')
  },
  {
    image: new URL('../../assets/robots/unitree g1edu.png', import.meta.url).href,
    name: 'Unitree G1edu',
    type: t('highlights.swiper.humanoid'),
    features: t('highlights.swiper.dexterous')
  }
];

// 获取图片URL
const getImageUrl = (imageName) => {
  return `/robots/${imageName}`;
};
</script>

<style scoped>
.robot-carousel {
  padding: 1rem;
  border: 1px solid #e1e5e9;
  border-radius: 12px;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.section-title {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 1rem;
  color: #2c3e50;
  font-weight: 700;
}

.carousel-container {
  height: 350px;
  border-radius: 16px;
  overflow: hidden;
  background: white;
}

.robot-swiper {
  width: 100%;
  height: 100%;
  position: relative;
}

.robot-slide {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}

.robot-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.3s ease;
}

.robot-card:hover {
  transform: translateY(-5px);
}

.robot-image {
  width: 100%;
  height: 200px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.robot-image img {
  width: 100%;
  height: 200px;
  object-fit: contain;
  transition: transform 0.5s ease;
}

.robot-card:hover .robot-image img {
  transform: scale(1.05);
}

.robot-info {
  width: 100%;
  padding: 0.5rem 0;
  text-align: center;
  background: white;
}

.robot-name {
  font-size: 1.2rem;
  margin-bottom: 0.2rem;
  font-weight: 600;
  line-height: 1.3;
}

.robot-type {
  font-size: 1.2rem;
  color: #3498db;
  margin-bottom: 0.2rem;
  font-weight: 500;
}

.robot-features {
  font-size: 1rem;
  color: #7f8c8d;
  font-style: italic;
  margin: 0;
}

/* Swiper导航按钮样式 */
.swiper-button-prev,
.swiper-button-next {
  color: #3498db;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.swiper-button-prev:hover,
.swiper-button-next:hover {
  color: rgb(0, 38, 255);
  transform: scale(1.1);
}

.swiper-button-prev::after,
.swiper-button-next::after {
  font-size: 1.2rem;
  font-weight: bold;
}

/* Swiper分页器样式 */
.swiper-pagination-bullet {
  width: 12px;
  height: 12px;
  opacity: 0.7;
  transition: all 0.3s ease;
}

.swiper-pagination-bullet-active {
  background: #3498db;
  opacity: 1;
  transform: scale(1.2);
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .carousel-container {
    height: 550px;
  }
  
  .robot-image {
    height: 300px;
  }
}

@media (max-width: 768px) {
  .robot-carousel {
  }
  
  .section-title {
    font-size: 2rem;
  }
  
  .carousel-container {
    height: 500px;
  }
  
  .robot-slide {
  }
  
  .robot-image {
    height: 250px;
  }
  
  .robot-name {
    font-size: 1.2rem;
  }
  
  .robot-type {
    font-size: 1.1rem;
  }
}

@media (max-width: 480px) {
  .container {
  }
  
  .carousel-container {
    height: 450px;
  }
  
  .robot-image {
    height: 200px;
  }
  
  .robot-info {
  }
  
  .robot-name {
    font-size: 1.3rem;
  }
  
  .swiper-button-prev,
  .swiper-button-next {
    display: none;
  }
}
</style>