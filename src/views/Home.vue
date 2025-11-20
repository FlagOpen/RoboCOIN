<template>
  <div class="home-container">
    <Header />
    <CoverVideo />
    <CoreData />
    <Highlights />
    <DatasetComparison />
    <AnnotationProcess />
    <Partners />
    <Citation />
    <Footer />

    <!-- 微信悬浮按钮 -->
    <div class="wechat-float-btn" @mouseenter="showQRCode = true" @mouseleave="showQRCode = false">
      <div class="wechat-icon">
        <img src="/wechat.svg" alt="微信">
      </div>
      <div v-if="showQRCode" class="qrcode-popup">
        <div class="qrcode-content">
          <img src="/qrcode.png" alt="微信二维码">
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import Header from '@/components/Header.vue'
import CoverVideo from '@/components/CoverVideo.vue'
import CoreData from '@/components/CoreData.vue'
import Highlights from '@/components/Highlights.vue'
import InteractiveCharts from '@/components/InteractiveCharts.vue'
import AnnotationProcess from '@/components/AnnotationProcess.vue'
import DatasetComparison from '@/components/DatasetComparison.vue'
import Partners from '@/components/Partners.vue'
import Footer from '@/components/Footer.vue'
import Citation from '@/components/Citation.vue'

export default {
  name: 'Home',
  components: {
    Header,
    CoverVideo,
    CoreData,
    Highlights,
    InteractiveCharts,
    AnnotationProcess,
    DatasetComparison,
    Partners,
    Citation,
    Footer
  },
  data() {
    return {
      showQRCode: false
    }
  }
}
</script>

<style scoped>
.home-container {
  width: 100%;
}

.wechat-float-btn {
  position: fixed;
  right: 30px;
  bottom: 30px;
  z-index: 1000;
  cursor: pointer;
}

.wechat-icon {
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: transform 0.3s ease;
}

.wechat-icon:hover {
  transform: scale(1.1);
}

.wechat-icon img {
  width: 60px;
  height: 60px;
}

.qrcode-popup {
  position: absolute;
  bottom: 70px;
  right: 0;
  background: white;
  border-radius: 8px;
  padding: 15px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  animation: fadeIn 0.3s ease;
}

.qrcode-popup::after {
  content: '';
  position: absolute;
  bottom: -8px;
  right: 20px;
  width: 0;
  height: 0;
  border-left: 8px solid transparent;
  border-right: 8px solid transparent;
  border-top: 8px solid white;
}

.qrcode-content img {
  width: 150px;
  height: 150px;
  display: block;
}

.qrcode-content p {
  margin: 10px 0 0 0;
  text-align: center;
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>