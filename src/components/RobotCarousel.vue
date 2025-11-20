<template>
  <section class="robot-carousel">
    <div class="container">
      <h2 class="section-title">Robot Type Introduction</h2>
      
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
// 引入Swiper组件和功能模块
import { Swiper, SwiperSlide } from 'swiper/vue';
import { Autoplay, Pagination, Navigation } from 'swiper/modules';
import 'swiper/css';
import 'swiper/css/pagination';
import 'swiper/css/navigation';

// 注册Swiper模块
const modules = [Autoplay, Pagination, Navigation];

// 机器人数据
const robots = [
  {
    // image: 'agilex cobot magic.png',
    image: new URL('../assets/robots/agilex cobot magic.png', import.meta.url).href,
    name: 'Agilex COBOT MAGIC',
    type: 'Dual Robot'
  },
  {
    // image: 'galaxea r1 lite.png',
    image: new URL('../assets/robots/galaxea r1 lite.png', import.meta.url).href,
    name: 'Galaxea R1 Lite',
    type: 'Dual Robot'
  },
  {
    // image: 'galbot g1.png',
    image: new URL('../assets/robots/galbot g1.png', import.meta.url).href,
    name: 'Galbot G1',
    type: 'Half-Humanoid'
  },
  {
    // image: 'tianqing a2.png',
    image: new URL('../assets/robots/tianqing a2.png', import.meta.url).href,
    name: 'Tianqing A2',
    type: 'Half-Humanoid'
  },
  {
    // image: 'realman aida l.png',
    image: new URL('../assets/robots/realman aida l.png', import.meta.url).href,
    name: 'Realman AIDA-L',
    type: 'Half-Humanoid'
  },
  {
    // image: 'ai2 alphabot 2.png',
    image: new URL('../assets/robots/ai2 alphabot 2.png', import.meta.url).href,
    name: 'AI2 Alphabot 2',
    type: 'Half-Humanoid',
    features: 'Dexterous Hand'
  },
  {
    // image: 'airbot mmk2.png',
    image: new URL('../assets/robots/airbot mmk2.png', import.meta.url).href,
    name: 'AIRBOT MMK2',
    type: 'Half-Humanoid',
    features: 'Dexterous Hand'
  },
  {
    // image: 'leju kuavo 4 pro.png',
    image: new URL('../assets/robots/leju kuavo 4 pro.png', import.meta.url).href,
    name: 'LEJU Kuavo 4 Pro',
    type: 'Humanoid',
    features: 'Dexterous Hand'
  },
  {
    // image: 'unitree g1edu.png',
    image: new URL('../assets/robots/unitree g1edu.png', import.meta.url).href,
    name: 'Unitree G1edu',
    type: 'Humanoid',
    features: 'Dexterous Hand'
  }
];

// 获取图片URL
const getImageUrl = (imageName) => {
  return `/robots/${imageName}`;
};
</script>

<style scoped>
.robot-carousel {
  padding: 4rem 0;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
}

.container {
  margin: 0 auto;
  padding: 0 2rem;
}

.section-title {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 2rem;
  color: #2c3e50;
  font-weight: 700;
}

.carousel-container {
  height: 500px;
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
  padding: 2rem;
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
  padding: 1.5rem;
  text-align: center;
  background: white;
}

.robot-name {
  font-size: 1.2rem;
  margin-bottom: 0.5rem;
  font-weight: 600;
  line-height: 1.3;
}

.robot-type {
  font-size: 1.2rem;
  color: #3498db;
  margin-bottom: 0.5rem;
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
  width: 50px;
  height: 50px;
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