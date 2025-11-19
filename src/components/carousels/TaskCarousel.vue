<template>
  <div class="video-selector-container">
    <!-- 左侧按钮面板 -->
    <div class="button-panel">
      <!-- 顶部X轴标签区域 -->
      <div class="axis-top">
        <div class="x-axis">
          <div class="x-ticks">
            <span v-for="(tick, index) in xAxisTicks" :key="index" class="x-tick">{{ tick }}</span>
          </div>
        </div>
      </div>

      <!-- 主内容区（Y轴刻度 + 按钮网格） -->
      <div class="main-content">
        <div class="y-axis">
          <div class="y-ticks">
            <span v-for="(tick, index) in yAxisTicks" :key="index" class="y-tick">{{ tick }}</span>
          </div>
        </div>
        
        <!-- 2x3按钮网格 -->
        <div class="button-grid">
          <button
            v-for="(video, index) in videoList"
            :key="index"
            class="video-button"
            :class="{ active: currentVideo === video.url }"
            @click="selectVideo(video)"
          >
            <span class="button-content">
              <span class="button-text">{{ video.name }}</span>
              <span class="button-hint">{{ t('highlights.demos.play') }}</span>
            </span>
          </button>
        </div>
      </div>
    </div>

    <!-- 右侧视频播放器 -->
    <div class="video-player-wrapper">
      <div class="title">Task Demos</div>
      <video
        ref="videoPlayer"
        :src="currentVideo"
        controls
        class="native-video-player"
        muted
      >
        Your browser does not support the video tag.
      </video>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import { ref } from 'vue';

const { t } = useI18n();

const videoList = [
  { name: t('highlights.demos.task.carry_parts'), url: new URL('../../assets/tasks/Low Rigid.mp4', import.meta.url).href },
  { name: t('highlights.demos.task.open_washer'), url: new URL('../../assets/tasks/Low Hinge.mp4', import.meta.url).href },
  { name: t('highlights.demos.task.flip_book'), url: new URL('../../assets/tasks/Low Deform.mp4', import.meta.url).href },
  { name: t('highlights.demos.task.arrange_books'), url: new URL('../../assets/tasks/High Rigid.mp4', import.meta.url).href },
  { name: t('highlights.demos.task.heat_food'), url: new URL('../../assets/tasks/High Hinge.mp4', import.meta.url).href },
  { name: t('highlights.demos.task.pass_pack'), url: new URL('../../assets/tasks/High Deform.mp4', import.meta.url).href }
];

const currentVideo = ref(new URL('../../assets/tasks/Low Rigid.mp4', import.meta.url).href);

// const xAxisTicks = ['Rigid\nObjects', 'Hinge\nObjects', 'Deformable\nObjects'];
// const yAxisTicks = ['Low\nAction Coordination', 'High\nAction Coordination'];
const xAxisTicks = [t('highlights.demos.object.rigid'), t('highlights.demos.object.hinged'), t('highlights.demos.object.deformable')];
const yAxisTicks = [t('highlights.demos.coordination.low'), t('highlights.demos.coordination.high')];
const videoPlayer = ref(null);

function selectVideo(video) {
  currentVideo.value = video.url;
  const player = videoPlayer.value;
  player.play().catch(error => {
    console.log('Auto-play was prevented:', error);
  });
}
</script>

<style scoped>
.video-selector-container {
  display: flex;
  width: 100%;
  padding: 10px 20px;
  gap: 20px;
  border: 1px solid #e1e5e9;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* 左侧按钮面板样式 */
.button-panel {
  flex: 0 0 55%;
  display: flex;
  flex-direction: column;
  border-radius: 12px;
  padding-bottom: 8px;
}

/* 顶部X轴区域 */
.axis-top {
  display: flex;
  flex-direction: column;
  margin-bottom: 15px;
}

.x-axis {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-left: 100px;
}

.x-ticks {
  display: flex;
  justify-content: space-around;
  flex: 1;
  gap: 0px;
}

/* 主内容区 */
.main-content {
  display: flex;
  flex: 1;
}

/* Y轴样式 */
.y-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  width: 100px;
  margin-right: 15px;
}

.y-ticks {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  height: 100%;
  gap: 20px;
}

/* 按钮网格 - 使用CSS Grid实现2x3布局 */
.button-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: 10px;
  flex: 1;
}

/* 美化后的按钮样式 */
.video-button {
  position: relative;
  font-size: 1.1rem;
  border: none;
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  box-shadow: 
    0 2px 4px rgba(0, 0, 0, 0.1),
    0 1px 2px rgba(0, 0, 0, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.8);
  color: #334155;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  overflow: hidden;
}

.video-button::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, 
    transparent, 
    rgba(255, 255, 255, 0.4), 
    transparent);
  transition: left 0.6s ease;
}

.video-button:hover::before {
  left: 100%;
}

.video-button:hover {
  background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
  box-shadow: 
    0 8px 25px rgba(0, 0, 0, 0.15),
    0 4px 6px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
  transform: translateY(-2px) scale(1.02);
  color: #1e293b;
}

.video-button.active {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  color: white;
  box-shadow: 
    0 6px 20px rgba(59, 130, 246, 0.4),
    0 3px 8px rgba(29, 78, 216, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  transform: translateY(0) scale(1);
}

.video-button.active:hover {
  background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
  box-shadow: 
    0 8px 25px rgba(37, 99, 235, 0.5),
    0 4px 10px rgba(30, 64, 175, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

.video-button:active {
  transform: translateY(1px) scale(0.98);
  box-shadow: 
    0 2px 5px rgba(0, 0, 0, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  transition: all 0.1s ease;
}

.button-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.button-text {
  font-size: 0.95rem;
  font-weight: 700;
}

.button-hint {
  font-size: 0.7rem;
  opacity: 0.7;
  font-weight: 500;
  transition: opacity 0.3s ease;
}

.video-button:hover .button-hint {
  opacity: 1;
}

.video-button.active .button-hint {
  opacity: 0.9;
}

/* 坐标轴标签样式优化 */
.x-tick {
  width: 30%;
  margin-left: 15px;
  font-size: 1rem;
  color: #475569;
  font-weight: 600;
  text-align: center;
  padding: 5px 10px;
  border-radius: 6px;
  white-space: pre-wrap;
}

.y-tick {
  font-size: 1rem;
  color: #475569;
  font-weight: 600;
  text-orientation: mixed;
  padding: 10px 5px;
  border-radius: 6px;
  white-space: pre-wrap;
  text-align: center;
}
/* 右侧视频播放器 */
.video-player-wrapper {
  flex: 1;
  min-width: 0;
  overflow: hidden;
}

.title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1e293b;
  text-align: center;
  margin-bottom: 15px;
}

.native-video-player {
  width: 100%;
  height: 270px;
  object-fit: cover;
  border-radius: 12px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .video-selector-container {
    flex-direction: column;
    gap: 15px;
  }
  
  .button-panel {
    flex: none;
    order: 2;
  }
  
  .button-grid {
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(3, 1fr);
  }
  
  .video-button {
    padding: 15px 10px;
    font-size: 1rem;
  }
}
</style>