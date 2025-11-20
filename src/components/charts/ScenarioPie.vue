<template>
  <div ref="chartContainer" style="width: 100%; height: 400px;"></div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, markRaw } from 'vue';
import { useI18n } from 'vue-i18n';
import * as echarts from 'echarts';

const { t } = useI18n();

// 图表容器和实例
const chartContainer = ref(null);
let chart = null;

// 颜色配置 - 每个类别一个主题色
const categoryColors = {
  residential: '#2E8B57', // 海绿色 - 住宅主题
  commercial: '#1E90FF',  // 道奇蓝 - 商业主题
  working: '#FF8C00'      // 暗橙色 - 工作主题
};

// 生成渐变色函数
const generateGradientColors = (baseColor, count) => {
  const colors = [];
  const baseRGB = hexToRgb(baseColor);
  
  for (let i = 0; i < count; i++) {
    const ratio = i / (count - 1);
    // 从深到浅渐变，保持色相不变，调整亮度
    const r = Math.round(baseRGB.r + (255 - baseRGB.r) * ratio * 0.6);
    const g = Math.round(baseRGB.g + (255 - baseRGB.g) * ratio * 0.6);
    const b = Math.round(baseRGB.b + (255 - baseRGB.b) * ratio * 0.6);
    
    colors.push(rgbToHex(r, g, b));
  }
  
  return colors.reverse(); // 反转使深色在前，浅色在后
};

// 16进制转RGB
const hexToRgb = (hex) => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
};

// RGB转16进制
const rgbToHex = (r, g, b) => {
  return '#' + [r, g, b].map(x => {
    const hex = x.toString(16);
    return hex.length === 1 ? '0' + hex : hex;
  }).join('');
};

// 外层大类数据
const outerData = [
  { 
    name: t('highlights.scenario.residential'), 
    value: 56242, 
    key: 'residential',
    children: [
      { name: t('highlights.scenario.kitchen'), value: 35827 },
      { name: t('highlights.scenario.bedroom'), value: 7381 },
      { name: t('highlights.scenario.living_room'), value: 7313 },
      { name: t('highlights.scenario.children_room'), value: 3844 },
      { name: t('highlights.scenario.home'), value: 1877 },
    ]
  },
  { 
    name: t('highlights.scenario.commercial'), 
    value: 18100, 
    key: 'commercial',
    children: [
      { name: t('highlights.scenario.restaurant'), value: 10582 },
      { name: t('highlights.scenario.courier_station'), value: 3839 },
      { name: t('highlights.scenario.supermarket'), value: 2393 },
      { name: t('highlights.scenario.amusement_park'), value: 1286 },
    ]
  },
  { 
    name: t('highlights.scenario.working'), 
    value: 36801, 
    key: 'working',
    children: [
      { name: t('highlights.scenario.factory'), value: 11243 },
      { name: t('highlights.scenario.office'), value: 10038 },
      { name: t('highlights.scenario.warehouse'), value: 6882 },
      { name: t('highlights.scenario.classroom'), value: 2831 },
      { name: t('highlights.scenario.laboratory'), value: 2727 },
      { name: t('highlights.scenario.laundry'), value: 2233 },
      { name: t('highlights.scenario.toll_booth'), value: 847 },
    ]
  }
];

// 当前选中类型和索引
const currentSelected = ref('Residential');
let currentIndex = 0;
let autoSwitchTimer = null;
const switchInterval = 3000;

// 初始化图表
const initChart = () => {
  if (!chartContainer.value) return;
  
  chart = markRaw(echarts.init(chartContainer.value));
  
  // 配置外层饼图颜色
  const outerPieColors = outerData.map(item => categoryColors[item.key]);
  
  // 配置选项
  const option = {
    title: {
      text: currentSelected.value,
      left: 'center',
      top: 'center',
      textStyle: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#333'
      }
    },
    tooltip: {
      trigger: 'item',
      formatter: '{a} <br/>{b}: {c} ({d}%)'
    },
    legend: {
      show: true,
      orient: 'vertical',
      left: 'left',
      data: outerData.map(item => item.name),
      textStyle: {
        fontSize: 12
      }
    },
    series: [
      {
        name: 'Type',
        type: 'pie',
        selectedMode: 'single',
        radius: ['30%', '50%'],
        label: {
          show: false,
        },
        emphasis: {
          label: {
            show: false,
            fontSize: 14,
            fontWeight: 'bold'
          },
        },
        data: outerData,
        itemStyle: {
          color: (params) => outerPieColors[params.dataIndex],
          borderColor: '#fff',
          borderWidth: 2,
        }
      },
      {
        name: 'Name',
        type: 'pie',
        radius: ['60%', '80%'],
        label: {
          show: true,
          formatter: '{b}\n{c}',
          fontSize: 12,
          fontWeight: 'normal'
        },
        labelLine: {
          show: true,
          length: 10,
          length2: 20
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold'
          },
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        },
        data: [],
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 2
        }
      }
    ]
  };

  chart.setOption(option);
  
  // 添加事件监听
  chart.on('click', (params) => {
    if (params.seriesIndex === 0) {
      handleCategoryClick(params.dataIndex);
    }
  });

  chart.on('mouseover', () => {
    stopAutoSwitch();
  });

  chart.on('mouseout', () => {
    startAutoSwitch();
  });

  // 默认显示第一个大类的子类
  if (outerData.length > 0) {
    switchToCategory(0);
  }

  startAutoSwitch();
};

// 处理分类点击
const handleCategoryClick = (index) => {
  stopAutoSwitch();
  switchToCategory(index);
};

// 切换到指定分类
const switchToCategory = (index) => {
  if (index < 0 || index >= outerData.length) return;
  
  const clickedCategory = outerData[index];
  currentSelected.value = clickedCategory.name;
  currentIndex = index;
  
  const innerData = clickedCategory.children || [];
  
  // 生成内层饼图的渐变色
  const baseColor = categoryColors[clickedCategory.key];
  const gradientColors = generateGradientColors(baseColor, innerData.length);
  
  // 为内层数据添加颜色
  const coloredInnerData = innerData.map((item, i) => ({
    ...item,
    itemStyle: {
      color: gradientColors[i]
    }
  }));
  
  chart.setOption({
    series: [
      {},
      { 
        data: coloredInnerData,
        itemStyle: {
          color: (params) => coloredInnerData[params.dataIndex].itemStyle.color
        }
      }
    ],
    title: {
      text: currentSelected.value
    }
  });
};

// 切换到下一个分类
const switchToNextCategory = () => {
  const nextIndex = (currentIndex + 1) % outerData.length;
  switchToCategory(nextIndex);
};

// 启动自动切换
const startAutoSwitch = () => {
  stopAutoSwitch();
  
  autoSwitchTimer = setInterval(() => {
    switchToNextCategory();
  }, switchInterval);
};

// 停止自动切换
const stopAutoSwitch = () => {
  if (autoSwitchTimer) {
    clearInterval(autoSwitchTimer);
    autoSwitchTimer = null;
  }
};

// 响应窗口大小变化
const handleResize = () => {
  if (chart) {
    chart.resize();
  }
};

// 页面可见性变化处理
const handleVisibilityChange = () => {
  if (document.hidden) {
    stopAutoSwitch();
  } else {
    startAutoSwitch();
  }
};

onMounted(() => {
  initChart();
  window.addEventListener('resize', handleResize);
  document.addEventListener('visibilitychange', handleVisibilityChange);
});

onUnmounted(() => {
  stopAutoSwitch();
  if (chart) {
    chart.dispose();
  }
  window.removeEventListener('resize', handleResize);
  document.removeEventListener('visibilitychange', handleVisibilityChange);
});
</script>