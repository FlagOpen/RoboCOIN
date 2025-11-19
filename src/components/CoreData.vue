<template>
  <section class="core-data">
    <div class="container">
      <h2 class="section-title">{{ $t('overview.title') }}</h2>
      <div class="core-description" v-html="$t('overview.paragraphs.1')"></div>
      <div class="core-description" v-html="$t('overview.paragraphs.2')"></div>
      <div class="core-description" v-html="$t('overview.paragraphs.3')"></div>
      <div class="stats-grid">
        <div class="stat-item">
          <!-- <img src="/robot.svg" alt="Robot Icon" class="stat-icon"> -->
          <div class="stat-number">{{ $t('stats.num_embodiments') }}</div>
          <div class="stat-label">{{ $t('overview.boxes.robot') }}</div>
        </div>
        <div class="stat-item">
          <!-- <img src="/trajectory.svg" alt="Robot Icon" class="stat-icon"> -->
          <div class="stat-number">{{ $t('stats.num_trajectories') }}</div>
          <div class="stat-label">{{ $t('overview.boxes.trajectory') }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{{ $t('stats.num_scenarios') }}</div>
          <div class="stat-label">{{ $t('overview.boxes.scenario') }}</div>
        </div>
        <div class="stat-item">
          <div class="stat-number">{{ $t('stats.num_tasks') }}</div>
          <div class="stat-label">{{ $t('overview.boxes.task') }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'CoreData'
}
</script>

<style scoped>
.core-data {
  padding: 3rem 5%;
  background-color: #f8f9fa;
}

.core-description {
  font-size: 1.2rem;
  margin-bottom: 2rem;
  line-height: 1.6;
  color: #2c3e50;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.section-title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 1rem;
  color: #2c3e50;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
}

.stat-icon {
  width: 100px;
  height: 100px;
}

.stat-item {
  text-align: center;
  padding: 1rem;
  background: white;
  border-radius: 8px;
  /* box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); */
  transition: transform 0.3s;
}

.stat-item:hover {
  transform: translateY(-5px);
}

.stat-number {
  font-size: 3rem;
  font-weight: bold;
  color: #2c3e50;
}

.stat-label {
  font-size: 1.2rem;
  color: #2c3e50;
}

.highlight {
  font-weight: bold;
  color: #0011ff;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .stat-number {
    font-size: 2.5rem;
  }
}
</style>