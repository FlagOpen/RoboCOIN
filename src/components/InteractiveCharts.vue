<template>
  <section class="interactive-charts">
    <div class="container">
      <h2 class="section-title">Interactive Data Statistics</h2>
      <div class="charts-grid">
        <div class="chart-container">
          <h3>Robot Type Distribution</h3>
          <div class="chart-placeholder">Pie Chart</div>
        </div>
        <div class="chart-container">
          <h3>Scenario Distribution</h3>
          <div class="chart-placeholder">Pie Chart</div>
        </div>
        <div class="chart-container">
          <h3>Action Distribution</h3>
          <div class="chart-placeholder">Bar Chart</div>
        </div>
        <div class="chart-container">
          <h3>Object Distribution</h3>
          <div class="chart-placeholder">Bar Chart</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'InteractiveCharts'
}
</script>

<style scoped>
.interactive-charts {
  padding: 4rem 0;
  background-color: #f8f9fa;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.section-title {
  text-align: center;
  font-size: 2rem;
  margin-bottom: 3rem;
  color: #2c3e50;
}

.charts-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 2rem;
}

.chart-container {
  background: white;
  padding: 2rem;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
}

.chart-container h3 {
  margin-bottom: 1.5rem;
  color: #2c3e50;
}

.chart-placeholder {
  height: 250px;
  background-color: #ecf0f1;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .charts-grid {
    grid-template-columns: 1fr;
  }
}
</style>