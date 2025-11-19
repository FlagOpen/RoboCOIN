<template>
  <div class="dataset-table-container">
    <h2 class="section-title">{{ t('comparison.title') }}</h2>
    <div class="description" v-html="t('comparison.desc')" />
    <div class="table-wrapper">
      <table class="dataset-table">
        <thead>
          <tr>
            <th v-for="header in headers" :key="header.key" :class="header.class">
              {{ header.text }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in datasetRows" :key="row.dataset" 
              :class="{
                'highlight-row': row.highlight,
                'gray-text': row.gray
              }">
            <td class="dataset-name">{{ row.dataset }}</td>
            <td>{{ row.arm }}</td>
            <td>{{ row.embodiment }}</td>
            <td>{{ row.trajectory }}</td>
            <td>{{ row.task }}</td>
            <td>{{ row.skill }}</td>
            <td>{{ row.dexterous ? '✓' : '✗' }}</td>
            <td>{{ row.annotation }}</td>
            <td>{{ row.collection }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { useI18n } from 'vue-i18n'

const { locale, t } = useI18n()

const headers = [
  { key: 'dataset', text: t('comparison.table.header.dataset'), class: 'dataset-header' },
  { key: 'arm', text: t('comparison.table.header.arm'), class: 'arm-header' },
  { key: 'embodiment', text: t('comparison.table.header.embodiment'), class: 'embodiment-header' },
  { key: 'trajectory', text: t('comparison.table.header.trajectory'), class: 'trajectory-header' },
  { key: 'task', text: t('comparison.table.header.task'), class: 'task-header' },
  { key: 'skill', text: t('comparison.table.header.skill'), class: 'skill-header' },
  { key: 'dexterous', text: t('comparison.table.header.dexterous'), class: 'dexterous-header' },
  { key: 'annotation', text: t('comparison.table.header.annotation'), class: 'annotation-header' },
  { key: 'collection', text: t('comparison.table.header.collection'), class: 'collection-header' }
]

const datasetRows = [
  {
    dataset: 'Pinto and Gupta',
    arm: t('comparison.table.content.dual'),
    embodiment: 1,
    trajectory: '50k',
    task: t('comparison.table.content.na'),
    skill: 1,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'RoboNet',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '162k',
    task: t('comparison.table.content.na'),
    skill: t('comparison.table.content.na'),
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'MT-Opt',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '800k',
    task: 12,
    skill: 1,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'BridgeData',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '7.2k',
    task: 71,
    skill: 4,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'BC-Z',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '26k',
    task: 100,
    skill: 3,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'RH20T',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '13k',
    task: 140,
    skill: 33,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'RoboSet',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '98k',
    task: 38,
    skill: 6,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.30human_70scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'BridgeData V2',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '60k',
    task: t('comparison.table.content.na'),
    skill: 13,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.85human_15scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'DROID',
    arm: t('comparison.table.content.single'),
    embodiment: 1,
    trajectory: '76k',
    task: t('comparison.table.content.na'),
    skill: 86,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.30human_70scripted'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'Open X-Embodiment',
    arm: t('comparison.table.content.single_dual'),
    embodiment: 22,
    trajectory: '1.4M',
    task: '160k',
    skill: 217,
    dexterous: false,
    annotation: t('comparison.table.content.no'),
    collection: t('comparison.table.content.dataset_aggregation'),
    highlight: false,
    gray: true
  },
  {
    dataset: 'RoboMIND',
    arm: t('comparison.table.content.single_dual'),
    embodiment: 4,
    trajectory: '107k',
    task: 479,
    skill: 38,
    dexterous: true,
    annotation: t('comparison.table.content.flat'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'AgiBot World Beta',
    arm: t('comparison.table.content.dual'),
    embodiment: 1,
    trajectory: '1M',
    task: 217,
    skill: 87,
    dexterous: false,
    annotation: t('comparison.table.content.flat'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'Open Galaxea',
    arm: t('comparison.table.content.dual'),
    embodiment: 1,
    trajectory: '50k',
    task: 150,
    skill: 58,
    dexterous: true,
    annotation: t('comparison.table.content.flat'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: false,
    gray: false
  },
  {
    dataset: 'RoboCOIN (Ours)',
    arm: t('comparison.table.content.dual'),
    embodiment: t('stats.num_embodiments'),
    trajectory: t('stats.num_trajectories'),
    task: t('stats.num_tasks'),
    skill: t('stats.num_actions'),
    dexterous: true,
    annotation: t('comparison.table.content.hierarchical'),
    collection: t('comparison.table.content.human_teleoperation'),
    highlight: true,
    gray: false
  }
]

</script>

<style scoped>

.dataset-table-container {
  padding: 2rem 12%;
  background-color: #f8f9fa;
}
.section-title {
  text-align: center;
  font-size: 2.5rem;
  margin-bottom: 2rem;
  color: #2c3e50;
}

.description {
  font-size: 1.2rem;
  line-height: 1.6;
  color: #2c3e50;
  margin-bottom: 1rem;
  text-align: center;
}

h2 {
  text-align: center;
  margin-bottom: 1.5rem;
  color: #2c3e50;
}

.table-wrapper {
  background-color: #ffffff;
  overflow-x: auto;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.dataset-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  min-width: 1000px;
}

.dataset-table th {
  background-color: #ffffff;
  padding: 12px 8px;
  text-align: center;
  font-weight: 600;
  color: #2c3e50;
  border-bottom: 2px solid #dee2e6;
}

.dataset-table td {
  padding: 10px 8px;
  text-align: center;
  border-bottom: 1px solid #e9ecef;
  vertical-align: middle;
}

.dataset-table tbody tr:hover {
  background-color: #ffffff;
}

/* 特殊行样式 */
.highlight-row {
  background-color: #ffebee !important; /* 红色高亮背景 */
  font-weight: 600;
}

.highlight-row:hover {
  background-color: #ffcdd2 !important;
}

.gray-text {
  color: #757575; /* 灰色文本 */
}

.dataset-name {
  text-align: left;
  font-weight: 500;
  padding-left: 12px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .dataset-table {
    font-size: 13px;
  }
  
  .dataset-table th,
  .dataset-table td {
    padding: 8px 6px;
  }
}

@media (max-width: 768px) {
  .dataset-table-container {
    padding: 1rem;
  }
  
  h2 {
    font-size: 1.3rem;
  }
  
  .dataset-table {
    font-size: 12px;
    min-width: 800px;
  }
}

/* 符号样式 */
.dataset-table td:contains("✓") {
  color: #4caf50; /* 绿色勾号 */
  font-weight: bold;
}

.dataset-table td:contains("✗") {
  color: #f44336; /* 红色叉号 */
}
</style>