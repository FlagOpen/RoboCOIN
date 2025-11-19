<template>
  <div class="annotation-container">
    <div class="left-panel">
      <div class="image-display">
        <!-- Static image display with fade transition -->
        <transition name="fade" mode="out-in">
          <img 
            :key="currentIndex"
            :src="currentAnnotation.image"
            :alt="`Frame ${currentAnnotation.image}`"
            class="current-image"
            @error="handleImageError"
          />
        </transition>
        
        <!-- Navigation controls -->
        <div class="navigation-controls">
          <button @click="prevImage" class="nav-btn prev-btn">❮</button>
          <button @click="toggleAutoPlay" class="control-btn play-pause-btn">
            {{ autoPlayEnabled ? '❚❚' : '▶' }}
          </button>
          <button @click="nextImage" class="nav-btn next-btn">❯</button>
        </div>
      </div>
      
      <!-- Frame information below the image -->
      <div class="frame-info">
        <p><strong>{{ t('highlights.pyramid.frame') }}:</strong> {{ currentAnnotation.index }} / 400</p>
      </div>
    </div>
    
    <div class="right-panel">
      <div class="annotation-content">
        <!-- Scene-level annotation -->
        <div class="annotation-section scene-level">
          <div class="annotation-header">
            <span class="color-indicator scene-indicator"></span>
            <h3>{{ t('highlights.pyramid.levels.trajectory') }}</h3>
          </div>
          <div class="annotation-body">
            <p class="scene-annotation">{{ sceneLevelAnnotation }}</p>
          </div>
        </div>
        
        <!-- Segment-level annotation -->
        <div class="annotation-section segment-level">
          <div class="annotation-header">
            <span class="color-indicator segment-indicator"></span>
            <h3>{{ t('highlights.pyramid.levels.segment') }}</h3>
          </div>
          <div class="annotation-body">
            <p class="segment-annotation">{{ currentAnnotation.segment }}</p>
          </div>
        </div>
        
        <!-- Frame-level annotation -->
        <div class="annotation-section frame-level">
          <div class="annotation-header">
            <span class="color-indicator frame-indicator"></span>
            <h3>{{ t('highlights.pyramid.levels.frame') }}</h3>
          </div>
          <div class="annotation-body">
            <div class="frame-annotation">
              <div class="arm-actions">
                <div class="arm-section left-arm">
                  <h4>Left Arm</h4>
                  <ul>
                    <li v-for="(action, index) in currentAnnotation.leftArmActions" :key="'left-' + index">
                      {{ action }}
                    </li>
                  </ul>
                </div>
                <div class="arm-section right-arm">
                  <h4>Right Arm</h4>
                  <ul>
                    <li v-for="(action, index) in currentAnnotation.rightArmActions" :key="'right-' + index">
                      {{ action }}
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import { ref, onMounted, onUnmounted, computed } from 'vue'

const { t } = useI18n();

// Scene-level annotation (fixed)
const sceneLevelAnnotation = ref('A basket and a pink peach are placed on a white table, the basket is on the left and the peach is on the right.')

// Current image index
const currentIndex = ref(0)

// Auto-play control
const autoPlayEnabled = ref(true)
let autoPlayTimer = null

// Get image URL
const getImageUrl = (imageName) => {
  return new URL(`../../assets/annotations/${imageName}`, import.meta.url).href
}

// Annotation data with enhanced structure
const annotationData = ref([
  { 
    image: getImageUrl('frame_00000.png'), 
    index: 1,
    segment: 'Right gripper catch basket.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper open; Right arm forward, high speed, acceleration increasing, gripper open',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: open'],
    rightArmActions: ['Direction: forward', 'Speed: high', 'Acceleration: increasing', 'Gripper: open']
  },
  { 
    image: getImageUrl('frame_00010.png'), 
    index: 50,
    segment: 'Right gripper catch basket.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper open; Right arm downward, low speed, acceleration decreasing, gripper open',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: open'],
    rightArmActions: ['Direction: down', 'Speed: low', 'Acceleration: decreasing', 'Gripper: open']
  },
  { 
    image: getImageUrl('frame_00020.png'), 
    index: 100,
    segment: 'Right gripper catch basket.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper open; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: open'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00030.png'), 
    index: 150,
    segment: 'Right gripper lift basket.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper open; Right arm upward, low speed, acceleration increasing, gripper closed',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: open'],
    rightArmActions: ['Direction: up', 'Speed: low', 'Acceleration: increasing', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00040.png'), 
    index: 200,
    segment: 'Left gripper catch peach.', 
    frame: 'Left arm downward, high speed, acceleration increasing, gripper open; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: down', 'Speed: high', 'Acceleration: increasing', 'Gripper: open'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00050.png'), 
    index: 250,
    segment: 'Left gripper catch peach.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper closed; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00055.png'), 
    index: 275,
    segment: 'Left gripper move peach over basket and release.', 
    frame: 'Left arm upward, high speed, acceleration increasing, gripper closed; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: up', 'Speed: high', 'Acceleration: increasing', 'Gripper: closed'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00060.png'), 
    index: 300,
    segment: 'Left gripper move peach over basket and release.', 
    frame: 'Left arm rightward, low speed, acceleration decreasing, gripper closed; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: right', 'Speed: low', 'Acceleration: decreasing', 'Gripper: closed'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00070.png'), 
    index: 350,
    segment: 'Left gripper move peach over basket and release.', 
    frame: 'Left arm stationary, speed stationary, acceleration constant, gripper open; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: open'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  },
  { 
    image: getImageUrl('frame_00080.png'), 
    index: 400,
    segment: 'Left gripper leave basket.', 
    frame: 'Left arm leftward, high speed, acceleration increasing, gripper open; Right arm stationary, speed stationary, acceleration constant, gripper closed',
    leftArmActions: ['Direction: left', 'Speed: high', 'Acceleration: increasing', 'Gripper: open'],
    rightArmActions: ['Direction: stationary', 'Speed: stationary', 'Acceleration: constant', 'Gripper: closed']
  }
])

// Current annotation data
const currentAnnotation = computed(() => {
  return annotationData.value[currentIndex.value] || {}
})

// Handle image loading errors
const handleImageError = (event) => {
  console.error(`Failed to load image: ${currentAnnotation.value.image}`)
}

// Previous image
const prevImage = () => {
  currentIndex.value = (currentIndex.value - 1 + annotationData.value.length) % annotationData.value.length
  if (autoPlayEnabled.value) {
    resetAutoPlay()
  }
}

// Next image
const nextImage = () => {
  currentIndex.value = (currentIndex.value + 1) % annotationData.value.length
  if (autoPlayEnabled.value) {
    resetAutoPlay()
  }
}

// Toggle auto-play
const toggleAutoPlay = () => {
  autoPlayEnabled.value = !autoPlayEnabled.value
  if (autoPlayEnabled.value) {
    startAutoPlay()
  } else {
    if (autoPlayTimer) {
      clearInterval(autoPlayTimer)
    }
  }
}

// Start auto-play
const startAutoPlay = () => {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer)
  }
  if (autoPlayEnabled.value) {
    autoPlayTimer = setInterval(() => {
      nextImage()
    }, 3000) // Switch every 3 seconds
  }
}

// Reset auto-play
const resetAutoPlay = () => {
  if (autoPlayEnabled.value) {
    startAutoPlay()
  }
}

// Component mounted
onMounted(() => {
  startAutoPlay()
})

// Component unmounted
onUnmounted(() => {
  if (autoPlayTimer) {
    clearInterval(autoPlayTimer)
  }
})
</script>

<style scoped>
.annotation-container {
  display: flex;
  width: 100%;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
  margin-top: 2rem;
}

.left-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  padding: 20px;
}

.right-panel {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background-color: #fff;
}

.image-display {
  position: relative;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.current-image {
  max-width: 100%;
  max-height: 100%;
  height: 300px;
  object-fit: contain;
  border-radius: 16px;
  margin-bottom: 70px;
}

.navigation-controls {
  position: absolute;
  bottom: 10px;
  left: 0;
  right: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
}

.nav-btn, .control-btn {
  background: rgba(0, 0, 0, 0.7);
  color: white;
  border: none;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 16px;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.nav-btn:hover, .control-btn:hover {
  background: rgba(0, 0, 0, 0.9);
  transform: scale(1.1);
}

.frame-info {
  text-align: center;
  padding: 12px;
  background-color: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.frame-info p {
  color: #555;
  font-size: 1.1em;
  font-weight: 500;
}

.annotation-content {
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.annotation-section {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.annotation-section:hover {
  transform: translateY(-2px);
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
}

.annotation-header {
  display: flex;
  align-items: center;
  padding: 12px 15px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.annotation-header h3 {
  margin: 0;
  font-size: 1.1em;
  font-weight: 600;
}

.color-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 10px;
}

.scene-level {
  border-left: 4px solid #e74c3c;
}

.scene-level .annotation-header {
  background-color: rgba(231, 76, 60, 0.1);
}

.scene-level .annotation-header h3 {
  color: #c0392b;
}

.scene-indicator {
  background-color: #e74c3c;
}

.segment-level {
  border-left: 4px solid #3498db;
}

.segment-level .annotation-header {
  background-color: rgba(52, 152, 219, 0.1);
}

.segment-level .annotation-header h3 {
  color: #2980b9;
}

.segment-indicator {
  background-color: #3498db;
}

.frame-level {
  border-left: 4px solid #2ecc71;
}

.frame-level .annotation-header {
  background-color: rgba(46, 204, 113, 0.1);
}

.frame-level .annotation-header h3 {
  color: #27ae60;
}

.frame-indicator {
  background-color: #2ecc71;
}

.annotation-body {
  padding: 10px 15px;
}

.scene-annotation, .segment-annotation {
  margin: 0;
  line-height: 1.2;
  color: #555;
  font-size: 0.95em;
}

.frame-annotation {
  margin: 0;
}

.arm-actions {
  display: flex;
  gap: 20px;
}

.arm-section {
  flex: 1;
}

p {
  color: #2c3e50;
}

.arm-section h4 {
  font-size: 1em;
  color: #2c3e50;
  font-weight: 600;
  border-bottom: 1px dashed #e0e0e0;
  padding-bottom: 5px;
}

.arm-section ul {
  margin: 0;
  padding-left: 0;
  list-style-type: none;
}

.arm-section li {
  font-size: 0.9em;
  color: #2c3e50;
}

.arm-section li:last-child {
  border-bottom: none;
}

/* Fade transition for image changes */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.4s ease;
}

.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

/* Responsive design */
@media (max-width: 768px) {
  .annotation-container {
    flex-direction: column;
    height: auto;
  }
  
  .image-display {
    height: 300px;
  }
  
  .arm-actions {
    flex-direction: column;
    gap: 15px;
  }
}
</style>