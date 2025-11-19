<template>
  <section class="highlights">
    <div class="container">
      <h1 class="section-title">{{ t('highlights.title') }}</h1>

      <div class="highlight-column">
        <div class="highlight-item">
          <div class="highlight-content">
            <h3>{{ t('highlights.robot.title') }}</h3>
            <p>{{ t('highlights.robot.desc') }}</p>
          </div>
          <RobotPie />
          <!-- <RobotCarousel /> -->
        </div>
        
        <div class="highlight-item">
          <div class="highlight-content">
            <h3>{{ t('highlights.scenario.title') }}</h3>
            <p>{{ t('highlights.scenario.desc') }}</p>
          </div>
          <ScenarioPie />
          <!-- <ScenarioCarousel /> -->
        </div>
      </div>

      <RobotCarousel />
      <ScenarioCarousel />
      
      <div class="highlight-item2">
        <div class="highlight-content">
          <h3>{{ t('highlights.task.title') }}</h3>
          <p>{{ t('highlights.task.desc') }}</p>
        </div>
        <ActionPie />
        <ObjectPie />
      </div>
      <TaskCarousel />
      
      <div class="highlight-item2">
        <div class="highlight-content">
          <h3>{{ t('highlights.pyramid.title') }}</h3>
          <p>{{ t('highlights.pyramid.desc') }}</p>
        </div>
        <CapabilityPyramid />
      </div>
    </div>
  </section>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import RobotPie from '@/components/charts/RobotPie.vue';
import ScenarioPie from '@/components/charts/ScenarioPie.vue';
import ActionPie from './charts/ActionPie.vue';
import ObjectPie from './charts/ObjectPie.vue';
import RobotCarousel from '@/components/carousels/RobotCarousel.vue';
import ScenarioCarousel from '@/components/carousels/ScenarioCarousel.vue';
import TaskCarousel from '@/components/carousels/TaskCarousel.vue';
import CapabilityPyramid from '@/components/carousels/CapabilityPyramid.vue';
const { t } = useI18n();
</script>

<style scoped>
.highlights {
  padding: 2rem 5%;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.section-title {
  text-align: center;
  font-size: 2.5rem;
  color: #2c3e50;
  margin-bottom: 2rem;
}

.highlight-column {
  display: flex;
  gap: 2rem;
  margin-bottom: 3rem;
  /* centering  */
  justify-content: center;
  align-items: center;
}

.highlight-item {
  width: 50%;
}

.highlight-item2 {
  margin-top: 2rem;
}

.highlight-image {
  flex: 1;
  height: 500px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
}

.highlight-content {
  flex: 1;
}

.highlight-content h3 {
  font-size: 1.8rem;
  color: #2c3e50;
}

.highlight-content p {
  font-size: 1.2rem;
  line-height: 1.6;
  color: #2c3e50;
}

p {
  color: #2c3e50;
}

@media (max-width: 768px) {
  .highlight-item {
    flex-direction: column;
    gap: 1.5rem;
  }
  
  .highlight-image {
    width: 100%;
    height: 200px;
  }
}
</style>