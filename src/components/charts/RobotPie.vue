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

// 定义每类机器人的主题色
const categoryColors = {
  dual: {
    primary: '#5470c6',
    gradient: ['#8ba0f7', '#5470c6', '#2c4cb0'] // 浅蓝 -> 主蓝 -> 深蓝
  },
  half_humanoid: {
    primary: '#91cc75',
    gradient: ['#b8e4a3', '#91cc75', '#6ab44c'] // 浅绿 -> 主绿 -> 深绿
  },
  humanoid: {
    primary: '#fac858',
    gradient: ['#fcdb8b', '#fac858', '#f7b325'] // 浅黄 -> 主黄 -> 深黄
  }
};

// 外层大类数据
const outerData = [
  { 
    name: t('highlights.robot.dual'), 
    value: 61149, 
    category: 'dual',
    children: [
      { name: 'Agilex Cobot Magic', value: 42454 },
      { name: 'Agilex Split Aloha', value: 9643 },
      { name: 'Galaxea R1 Lite', value: 8800 },
    ]
  },
  { 
    name: t('highlights.robot.half_humanoid'), 
    value: 60043, 
    category: 'half_humanoid',
    children: [
      { name: 'Realman-AIDA-L', value: 20067 },
      { name: 'Tianqing A2-D', value: 16170 },
      { name: 'AI2 AlphaBot 2', value: 15916 },
      { name: 'AI2 AlphaBot 1s', value: 15000 },
      { name: 'Galbot g1', value: 11316 },
      { name: 'Airbot MMK2', value: 5901 },
      { name: 'Tianqing A2', value: 2018 },
      { name: 'Leju Kuavo 4 LB', value: 1440 },
      { name: 'Realman Rs-02', value: 1297 },
      { name: 'Realman Rs-01', value: 763 },
    ]
  },
  { 
    name: t('highlights.robot.humanoid'), 
    value: 22499, 
    category: 'humanoid',
    children: [
      { name: 'Leju Kuavo 4 Pro', value: 21742 },
      { name: 'Unitree G1edu-u3', value: 10885 }
    ]
  }
];

// 当前选中类型
const currentSelected = ref('Dual');
// 定时器引用
let autoSwitchTimer = null;
// 当前高亮索引
let currentHighlightIndex = 0;
// 切换间隔（毫秒）
const switchInterval = 3000;

// 生成渐变颜色数组
const generateGradientColors = (baseColor, count) => {
  const colors = [];
  const gradientStops = baseColor.gradient;
  
  if (count <= gradientStops.length) {
    // 如果颜色数量少于渐变颜色数量，直接返回对应的颜色
    return gradientStops.slice(0, count);
  }
  
  // 生成平滑渐变
  for (let i = 0; i < count; i++) {
    const progress = i / (count - 1);
    if (progress <= 0.5) {
      // 前半段使用前两个颜色的渐变
      const subProgress = progress * 2;
      colors.push(interpolateColor(gradientStops[0], gradientStops[1], subProgress));
    } else {
      // 后半段使用后两个颜色的渐变
      const subProgress = (progress - 0.5) * 2;
      colors.push(interpolateColor(gradientStops[1], gradientStops[2], subProgress));
    }
  }
  
  return colors;
};

// 颜色插值函数
const interpolateColor = (color1, color2, factor) => {
  const hex = (color) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  };

  const rgb1 = hex(color1);
  const rgb2 = hex(color2);
  
  if (!rgb1 || !rgb2) return color1;
  
  const result = {
    r: Math.round(rgb1.r + (rgb2.r - rgb1.r) * factor),
    g: Math.round(rgb1.g + (rgb2.g - rgb1.g) * factor),
    b: Math.round(rgb1.b + (rgb2.b - rgb1.b) * factor)
  };
  
  return `#${((1 << 24) + (result.r << 16) + (result.g << 8) + result.b).toString(16).slice(1)}`;
};

// 初始化图表
const initChart = () => {
  if (!chartContainer.value) return;
  
  chart = markRaw(echarts.init(chartContainer.value));
  
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
      formatter: (params) => {
        if (params.seriesIndex === 0) {
          return `${params.name}<br/>数量: ${params.value}<br/>占比: ${params.percent}%`;
        } else {
          return `${params.name}<br/>数量: ${params.value}`;
        }
      }
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
          show: false
        },
        labelLine: {
          show: false
        },
        data: outerData.map(item => ({
          ...item,
          itemStyle: {
            color: categoryColors[item.category].primary
          }
        })),
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 2
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
          length2: 15
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 12,
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
  
  // 添加点击事件监听
  chart.on('click', (params) => {
    if (params.seriesIndex === 0) {
      handleCategoryClick(params.dataIndex);
    }
  });

  // 鼠标悬停时暂停自动切换
  chart.on('mouseover', () => {
    stopAutoSwitch();
  });

  // 鼠标移出时恢复自动切换
  chart.on('mouseout', () => {
    startAutoSwitch();
  });

  // 默认显示第一个大类的子类
  if (outerData.length > 0) {
    switchToCategory(0);
  }

  // 启动自动切换
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
  currentHighlightIndex = index;
  
  // 对子项按值排序，确保渐变颜色正确分配
  const innerData = [...clickedCategory.children].sort((a, b) => b.value - a.value);
  const gradientColors = generateGradientColors(categoryColors[clickedCategory.category], innerData.length);
  
  // 为每个子项分配颜色
  const coloredInnerData = innerData.map((item, i) => ({
    ...item,
    itemStyle: {
      color: gradientColors[i]
    }
  }));
  
  // 更新图表配置
  const currentOption = chart.getOption();
  
  // 修复：外层饼图保持主题色，使用echarts的高亮效果而不是手动修改颜色
  currentOption.series[0].data = outerData.map(item => ({
    ...item,
    itemStyle: {
      color: categoryColors[item.category].primary
    }
  }));
  
  currentOption.series[1].data = coloredInnerData;
  currentOption.title[0].text = currentSelected.value;
  chart.setOption(currentOption);
  
  // 高亮显示当前选中的大类 - 使用echarts内置高亮而不是修改颜色
  chart.dispatchAction({
    type: 'highlight',
    seriesIndex: 0,
    dataIndex: index
  });
  
  // 取消其他项的高亮
  for (let i = 0; i < outerData.length; i++) {
    if (i !== index) {
      chart.dispatchAction({
        type: 'downplay',
        seriesIndex: 0,
        dataIndex: i
      });
    }
  }
};

// 切换到下一个分类
const switchToNextCategory = () => {
  const nextIndex = (currentHighlightIndex + 1) % outerData.length;
  switchToCategory(nextIndex);
};

// 启动自动切换
const startAutoSwitch = () => {
  stopAutoSwitch(); // 确保没有重复的定时器
  
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

onMounted(() => {
  initChart();
  window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
  // 清理资源
  stopAutoSwitch();
  if (chart) {
    chart.dispose();
  }
  window.removeEventListener('resize', handleResize);
});
</script>