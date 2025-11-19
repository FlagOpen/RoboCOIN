<template>
  <section class="cover-video">
    <div class="video-container">
      <!-- 使用video标签替换占位符 -->
      <video 
        ref="videoPlayer"
        class="video-element"
        :poster="videoPoster"
        autoplay
        muted
        loop
        @loadeddata="handleVideoLoaded"
      >
        <source src="/overview.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      
      <!-- 保留覆盖文字 -->
      <div class="video-overlay">
        <h1 class="video-title">
          <span style="color: #ff7f00;">Robo</span><span style="color: #0099ff;">CO</span><span style="color: #00c766;">IN</span>:
          An Open-Sourced Bimanual <span style="color: #ff7f00;">Robo</span>tic<br>
          Data <span style="color: #0099ff;">CO</span>llection for <span style="color: #00c766;">IN</span>tegrated Manipulation
        </h1>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'CoverVideo',
  data() {
    return {
      videoPoster: '', // 可以设置为封面图片路径
      isVideoLoaded: false
    }
  },
  methods: {
    handleVideoLoaded() {
      this.isVideoLoaded = true;
    }
  },
  mounted() {
    // 组件挂载后可以执行视频相关初始化
    // 如果需要动态设置封面，可以在这里进行
    this.videoPoster = ''; // 设置为您的封面图片路径，或留空使用视频第一帧
  }
}
</script>

<style scoped>
.cover-video {
  margin-top: 60px; /* 为固定头部留出空间 */
  height: 96vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #1a2a3a, #2c3e50);
  position: relative;
}

.video-container {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
}

/* 视频元素样式 */
.video-element {
  width: 100%;
  height: 100%;
  object-fit: cover; /* 确保视频覆盖整个容器 */
  display: block;
}

.video-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: white;
  padding: 2rem;
  background: rgba(0, 0, 0, 0.3); /* 半透明背景增强文字可读性 */
}

.video-title {
  font-size: 2.5rem;
  font-weight: 700;
  line-height: 1.3;
  max-width: 1600px;
  margin: 0 auto;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

@media (max-width: 768px) {
  .video-title {
    font-size: 1.8rem;
  }
  
  .cover-video {
    height: 50vh; /* 移动端调整高度 */
  }
}
</style>