<template>
  <div ref="chartContainer" style="width: 100%; height: 250px;"></div>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import { ref, onMounted, onUnmounted, markRaw } from 'vue';
import * as echarts from 'echarts';

const { t } = useI18n();

const chartContainer = ref(null);
let chart = null;

// 原始数据
const actionData = {
  low: [
{ name: 'grasp', value: 114469 },
{ name: 'pick', value: 113147 },
{ name: 'place', value: 108248 },
{ name: 'press', value: 4471 },
{ name: 'pour', value: 3423 },
{ name: 'push', value: 3147 },
{ name: 'receive', value: 2582 },
{ name: 'wipe', value: 2223 },
{ name: 'pull', value: 1710 },
{ name: 'stir', value: 885 },
{ name: 'open', value: 807 },
{ name: 'take', value: 779 },
{ name: 'put', value: 779 },
{ name: 'move', value: 779 },
{ name: 'scoop', value: 500 },
{ name: 'close', value: 413 },
{ name: 'flip', value: 210 },
{ name: 'walk', value: 200 },
{ name: 'insert', value: 125 },
{ name: 'rotate', value: 99 },
{ name: 'lower', value: 97 },
{ name: 'hold', value: 93 },
{ name: 'pullapart', value: 16 },
{ name: 'turn', value: 16 },
{ name: 'together', value: 16 },
  ],
  high: [
{ name: 'grasp', value: 43964 },
{ name: 'place', value: 38116 },
{ name: 'pick', value: 37616 },
{ name: 'pull', value: 6606 },
{ name: 'fold', value: 5117 },
{ name: 'flod', value: 3182 },
{ name: 'pour', value: 2550 },
{ name: 'push', value: 1711 },
{ name: 'close', value: 1506 },
{ name: 'twist', value: 1501 },
{ name: 'up', value: 1441 },
{ name: 'zip', value: 1441 },
{ name: 'press', value: 1227 },
{ name: 'insert', value: 758 },
{ name: 'sweep', value: 703 },
{ name: 'cut', value: 656 },
{ name: 'move', value: 655 },
{ name: 'takeout', value: 507 },
{ name: 'hold', value: 434 },
{ name: 'wipe', value: 428 },
{ name: 'flip', value: 392 },
{ name: 'unfold', value: 317 },
{ name: 'open', value: 298 },
{ name: 'clip', value: 78 },
{ name: 'cover', value: 50 },
  ]
};

// 处理数据：分别按值降序排序，并提取名称和数值数组
const processData = () => {
  const sortedLow = [...actionData.low].sort((a, b) => b.value - a.value);
  const sortedHigh = [...actionData.high].sort((a, b) => b.value - a.value);
  
  return {
    lowCategories: sortedLow.map(item => item.name),
    lowValues: sortedLow.map(item => item.value),
    highCategories: sortedHigh.map(item => item.name),
    highValues: sortedHigh.map(item => item.value)
  };
};

const initChart = () => {
  if (!chartContainer.value) return;
  
  chart = markRaw(echarts.init(chartContainer.value));
  const { lowCategories, lowValues, highCategories, highValues } = processData();

  const option = {
    title: {
      left: 'center',
      textStyle: { fontSize: 16 }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' }
    },
    // 关键配置：定义左右两个图表区域
    grid: [
      { // 左侧网格（低协作图表）
        left: '0%',
        right: '50%', // 左侧图表占据约一半少一点的空间，为中间留出间隙
        top: '5%',
        bottom: '10%',
        containLabel: true
      },
      { // 右侧网格（高协作图表）
        left: '51%', // 从中间偏右开始
        right: '0%',
        top: '5%',
        bottom: '15%',
        containLabel: true
      }
    ],
    yAxis: [
      { // 左侧x轴
        type: 'log',
        gridIndex: 0, // 关联到第一个grid
        position: 'bottom'
      },
      { // 右侧x轴
        type: 'log',
        gridIndex: 1, // 关联到第二个grid
        position: 'bottom'
      }
    ],
    xAxis: [
      { // 左侧y轴（低协作动作名称）
        type: 'category',
        gridIndex: 0,
        name: t('highlights.task.coordination.low'),
        nameLocation: 'middle',
        nameGap: 70,
        data: lowCategories,
        axisLabel: { color: '#ffa586', rotate: 90 },
      },
      { // 右侧y轴（高协作动作名称）
        type: 'category',
        gridIndex: 1,
        name: t('highlights.task.coordination.high'),
        nameLocation: 'middle',
        nameGap: 70,
        data: highCategories,
        axisLabel: { color: '#e47869', rotate: 90 },
      }
    ],
    series: [
      {
        name: t('highlights.task.coordination.low'),
        type: 'bar',
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: lowValues,
        itemStyle: { color: '#ffa586' },
      },
      {
        name: t('highlights.task.coordination.high'),
        type: 'bar',
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: highValues,
        itemStyle: { color: '#e47869' },
      }
    ]
  };

  chart.setOption(option);
};

const handleResize = () => {
  if (chart) chart.resize();
};

onMounted(() => {
  initChart();
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  if (chart) chart.dispose();
  window.removeEventListener('resize', handleResize);
});
</script>