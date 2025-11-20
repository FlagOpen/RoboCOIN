<template>
  <footer class="star-footer">
    <!-- 星空背景 -->
    <div class="stars">
      <div v-for="(star, index) in stars" :key="index" 
           class="star" :style="starStyle(star)"></div>
    </div>
    
    <!-- 内容区域 -->
    <div class="footer-content">
      <h3 class="open-source-date">{{ $t('footer.title') }}</h3>

      <div class="data-stats">
        <div class="stat-item">{{ $t('footer.boxes.robot') }}</div>
        <div class="stat-item">{{ $t('footer.boxes.data') }}</div>
        <div class="stat-item">{{ $t('footer.boxes.toolchain') }}</div>
      </div>
      
      <!-- 修改后的左右排版打字机效果区域 -->
      <div class="typewriter-section">
        <div class="typewriter-container">
          <div class="text-left">
            <p class="get-started-text">{{ $t('footer.start') }}</p>
          </div>
          <div class="command-right">
            <div class="command-display">
              <code class="typewriter-command">{{ typedCommand }}</code>
              <span class="typing-cursor" :class="{ blinking: isTyping }">|</span>
            </div>
          </div>
        </div>
      </div>
      
      <p class="invitation-text">
        {{ $t('footer.text') }}
      </p>
      
      <div class="qrcode-section">
        <div class="qrcode-container">
          <img class="qrcode-placeholder" src="../assets/qrcode.png" alt="QR Code" />
        </div>
      </div>
    </div>
    <div class="footer-copyright">
      {{ $t('footer.copyright') }}
    </div>
  </footer>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'

// 生成星星数据
const stars = ref([])

// 语言切换
const currentLanguage = ref('zh')
const switchLanguage = (lang) => {
  currentLanguage.value = lang
  // 这里可以添加语言切换逻辑
}

// 打字机效果相关变量
const commandText = 'pip install robocoin'
const typedCommand = ref('')
const isTyping = ref(false)
const typingSpeed = 100 // 打字速度（毫秒）
const typingTimer = ref(null)
const loopInterval = ref(null) // 循环定时器
const loopDelay = 6000 // 5秒循环一次

// 打字机效果实现
const startTypewriter = async () => {
  isTyping.value = true
  typedCommand.value = ''
  
  // 等待DOM更新
  await nextTick()
  
  for (let i = 0; i < commandText.length; i++) {
    // 模拟打字延迟
    await new Promise(resolve => {
      typingTimer.value = setTimeout(() => {
        typedCommand.value += commandText[i]
        resolve()
      }, typingSpeed)
    })
  }
  
  // 打字完成后保持光标闪烁
  isTyping.value = false
}

// 启动循环打字效果
const startLoopTyping = () => {
  // 先执行一次打字效果
  startTypewriter()
  
  // 设置循环，每5秒重新执行一次
  loopInterval.value = setInterval(() => {
    startTypewriter()
  }, loopDelay)
}

// 生成随机星星
const generateStars = () => {
  const starCount = 100
  const newStars = []
  
  for (let i = 0; i < starCount; i++) {
    newStars.push({
      id: i,
      top: Math.random() * 100 + '%',
      left: Math.random() * 100 + '%',
      size: Math.random() * 3 + 1,
      delay: Math.random() * 5,
      duration: Math.random() * 3 + 2
    })
  }
  
  stars.value = newStars
}

// 星星样式
const starStyle = (star) => {
  return {
    top: star.top,
    left: star.left,
    width: star.size + 'px',
    height: star.size + 'px',
    animationDelay: star.delay + 's',
    animationDuration: star.duration + 's'
  }
}

// 组件挂载时生成星星并启动循环打字机效果
onMounted(() => {
  generateStars()
  // 延迟启动打字机效果，让页面先加载完成
  setTimeout(() => {
    startLoopTyping()
  }, 1000)
})

// 组件卸载前清除定时器
onUnmounted(() => {
  if (typingTimer.value) {
    clearTimeout(typingTimer.value)
  }
  if (loopInterval.value) {
    clearInterval(loopInterval.value)
  }
})
</script>

<style scoped>
/* 语言切换器样式 */
.language-switcher {
  position: absolute;
  top: 20px;
  right: 20px;
  z-index: 10;
  display: flex;
  gap: 10px;
}

.lang-btn {
  padding: 8px 16px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

.lang-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.lang-btn.active {
  background: rgba(100, 150, 255, 0.6);
  border-color: rgba(100, 150, 255, 0.8);
}

/* 其余样式保持不变 */
.star-footer {
  position: relative;
  background: linear-gradient(180deg, #1E40AF 0%, #0b0e25 100%);
  color: white;
  padding: 60px 20px 40px;
  text-align: center;
  overflow: hidden;
}

/* 星星效果 */
.stars {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.star {
  position: absolute;
  background-color: white;
  border-radius: 50%;
  animation: twinkle 4s infinite ease-in-out;
}

@keyframes twinkle {
  0%, 100% { 
    opacity: 0.3; 
    transform: scale(0.8);
  }
  50% { 
    opacity: 1; 
    transform: scale(1.2);
  }
}

/* 内容布局 */
.footer-content {
  position: relative;
  z-index: 1;
  max-width: 1200px;
  margin: 0 auto;
}

.open-source-date {
  font-size: 2.2rem;
  font-weight: bold;
  margin-bottom: 40px;
  color: #ffffff;
  text-shadow: 0 0 10px rgba(100, 150, 255, 0.5);
}

.data-stats {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 30px;
  margin-bottom: 40px;
}

.stat-item {
  font-size: 1.5rem;
  font-weight: 600;
  padding: 15px 25px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

/* 修改后的左右排版打字机效果区域样式 */
.typewriter-section {
  margin: 40px 0;
  padding: 30px;
  /* background: rgba(255, 255, 255, 0.05); */
  border-radius: 12px;
  /* border: 1px solid rgba(255, 255, 255, 0.1); */
  backdrop-filter: blur(10px);
}

.typewriter-container {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 20px;
  flex-wrap: wrap;
}

.text-left {
  flex: 0 0 auto;
}

.get-started-text {
  font-size: 1.4rem;
  margin: 0;
  color: #e0e0ff;
  opacity: 0.9;
  white-space: nowrap;
}

.command-right {
  flex: 0 0 auto;
}

.command-display {
  display: inline-flex;
  align-items: center;
  background: rgba(0, 0, 0, 0.3);
  padding: 15px 25px;
  border-radius: 8px;
  border: 1px solid rgba(100, 150, 255, 0.3);
  position: relative;
  min-height: 60px;
  min-width: 400px;
  justify-content: center;
}

.typewriter-command {
  font-family: 'Courier New', monospace;
  font-size: 1.3rem;
  font-weight: 600;
  color: #4fc3f7;
  background: linear-gradient(45deg, #4fc3f7, #29b6f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 1px;
}

.typing-cursor {
  font-size: 1.5rem;
  color: #4fc3f7;
  margin-left: 2px;
  opacity: 1;
}

.typing-cursor.blinking {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

.invitation-text {
  font-size: 1rem;
  line-height: 1.6;
  margin: 0 auto 10px;
  color: #e0e0ff;
  opacity: 0.9;
  max-width: 1000px;
}

.qrcode-section {
  margin-bottom: 10px;
}

.qrcode-container {
  display: inline-block;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  padding: 10px;
}

.qrcode-placeholder {
  width: 150px;
  height: 150px;
  border-radius: 8px;
}

.footer-copyright {
  margin-top: 30px;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.6);
  opacity: 0.8;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .star-footer {
    padding: 40px 15px 30px;
  }
  
  .open-source-date {
    font-size: 1.8rem;
    margin-bottom: 30px;
  }
  
  .data-stats {
    gap: 15px;
    margin-bottom: 30px;
  }
  
  .stat-item {
    font-size: 1.2rem;
    padding: 12px 20px;
  }
  
  .typewriter-section {
    margin: 30px 0;
    padding: 20px;
  }
  
  .typewriter-container {
    flex-direction: column;
    gap: 15px;
  }
  
  .get-started-text {
    font-size: 1.2rem;
  }
  
  .command-display {
    min-width: 250px;
    padding: 12px 20px;
  }
  
  .typewriter-command {
    font-size: 1.1rem;
  }
  
  .invitation-text {
    font-size: 0.9rem;
    margin-bottom: 30px;
  }
  
  .qrcode-placeholder {
    width: 120px;
    height: 120px;
  }
}

/* 小屏幕手机适配 */
@media (max-width: 480px) {
  .command-display {
    min-width: 200px;
    padding: 10px 15px;
  }
  
  .typewriter-command {
    font-size: 1rem;
  }
  
  .data-stats {
    flex-direction: column;
    align-items: center;
  }
}
</style>