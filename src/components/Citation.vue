<template>
  <div class="citation-component">
    <div class="citation-content">
      <button 
        class="copy-button" 
        @click="copyCitation"
        title="复制BibTeX引用"
      >
        <span class="copy-icon">📋</span>
      </button>
      
      <div class="citation-text" ref="citationText">
        <pre class="bibtex-code">{{ bibtexCitation }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

// 作者信息（已移除数字和符号）
const authors = [
  "Shihan Wu", "Xuecheng Liu", "Shaoxuan Xie", "Pengwei Wang", "Xinghang Li",
  "Bowen Yang", "Zhe Li", "Kai Zhu", "Hongyu Wu", "Yiheng Liu", "Zhaoye Long", "Yue Wang",
  "Chong Liu", "Dihan Wang", "Ziqiang Ni", "Xiang Yang", "You Liu", "Ruoxuan Feng", "Runtian Xu",
  "Lei Zhang", "Denghang Huang", "Chenghao Jin", "Anlan Yin", "Xinlong Wang", "Zhenguo Sun",
  "Mengfei Du", "Mingyu Cao", "Xiansheng Chen", "Hongyang Cheng", "Xiaojie Zhang", "Junkai Zhao",
  "Cheng Chi", "Sixiang Chen", "Huaihai Lyu", "Xiaoshuai Hao", "Yankai Fu", "Yequan Wang", "Bo Lei",
  "Dong Liu", "Xi Yang", "Yance Jiao", "Tengfei Pan", "Yunyan Zhang", "Songjing Wang", "Ziqian Zhang",
  "Xu Liu", "Caowei Meng", "Zhizheng Zhang", "He Wang", "Hang Zhao", "Jiyang Gao", "Song Wang",
  "Xiaokun Leng", "Zhiqiang Xie", "Zhenzhen Zhou", "Peng Huang", "Wu Yang", "Liaodong Guo",
  "Yichao Zhu", "Suibing Zheng", "Hao Cheng", "Xinmin Ding", "Yang Yue","Huanqian Wang", 
  "Chi Chen", "Jingrui Pang", "YuXi Qian", "Haoran Geng", "Lianli Gao", "Haiyuan Li", "Bin Fang",
  "Gao Huang", "Hao Dong", "Yadong Mu", "Di Hu", "Hao Zhao",
  "Shanghang Zhang", "Yonghua Lin", "Zhongyuan Wang", "Guocai Yao"
]

// 计算属性：生成BibTeX格式的引用
const bibtexCitation = computed(() => {
  // 处理作者列表，最后一个名字前用"and"连接
  let authorsFormatted;
  if (authors.length === 0) {
    authorsFormatted = "";
  } else if (authors.length === 1) {
    authorsFormatted = authors[0];
  } else if (authors.length === 2) {
    authorsFormatted = authors.join(' and ');
  } else {
    authorsFormatted = authors.slice(0, -1).join(', ') + ' and ' + authors.slice(-1);
  }
  
  return `@article{RoboCOINReport,
  author = {${authorsFormatted}},
  title = {RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation},
  year = {2025},
  url = {https://github.com/FlagOpen/RoboCOIN}
}`
})

// 复制引用到剪贴板
const copyCitation = async () => {
  const citation = bibtexCitation.value
  
  try {
    // 使用现代剪贴板API
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(citation)
    } else {
      // 回退方案：使用传统方法
      const textArea = document.createElement('textarea')
      textArea.value = citation
      textArea.style.position = 'fixed'
      textArea.style.left = '-999999px'
      textArea.style.top = '-999999px'
      document.body.appendChild(textArea)
      textArea.focus()
      textArea.select()
      const successful = document.execCommand('copy')
      document.body.removeChild(textArea)
      
      if (!successful) {
        throw new Error('复制失败')
      }
    }
  } catch (err) {
    console.error('复制失败:', err)
    alert('复制失败，请手动选择文本复制')
  }
}
</script>

<style scoped>
.citation-component {
  position: relative;
  /* background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); */
  /* border: 1px solid #e0e0e0; */
  /* border-radius: 12px; */
  padding: 24px 10%;
  margin: 20px ;
  font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
}

.citation-content {
  position: relative;
  display: flex;
  flex-direction: column;
}

/* 简化复制按钮样式 */
.copy-button {
  position: absolute;
  top: -10px;
  right: -10px;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f0f0f0;
  color: #666;
  border: 1px solid #ddd;
  border-radius: 50%;
  cursor: pointer;
  font-size: 1.2rem;
  z-index: 10;
}

.copy-button:hover {
  background: #e0e0e0;
}

.citation-text {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 20px;
  position: relative;
  overflow-x: auto;
}

.bibtex-code {
  margin: 0;
  color: #2c3e50;
  font-size: 1.0rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Courier New', 'Monaco', 'Menlo', monospace;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .citation-component {
    padding: 16px;
    margin: 16px 0;
  }
  
  .citation-text {
    padding: 15px;
  }
  
  .bibtex-code {
    font-size: 0.8rem;
  }
  
  .copy-button {
    width: 36px;
    height: 36px;
    font-size: 1.1rem;
    top: -8px;
    right: -8px;
  }
}

@media (max-width: 480px) {
  .citation-component {
    padding: 12px;
  }
  
  .citation-text {
    padding: 12px;
  }
  
  .bibtex-code {
    font-size: 0.75rem;
  }
}

/* 打印样式 */
@media print {
  .citation-component {
    background: white;
    border: 1px solid #ccc;
    box-shadow: none;
    page-break-inside: avoid;
  }
  
  .copy-button {
    display: none;
  }
  
  .bibtex-code {
    color: black;
  }
}
</style>