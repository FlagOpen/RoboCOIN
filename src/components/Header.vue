<template>
  <header class="header">
    <div class="logo">
      <img class="logo-image" src="/logos/robocoin.png" alt="RoboCOIN Logo" />
    </div>
    <div class="header-right">
      <nav class="nav-links">
        <!-- <a href="#" class="nav-link">Download</a>
        <a href="#" class="nav-link">GitHub</a>
        <a href="#" class="nav-link">Visualization</a>
        <a href="#" class="nav-link">About Us</a> -->
        <div class="nav-link">
          <img class="nav-icon" src="/icons/pdf.svg" alt="Technical Report Icon" />
          <a class="nav-item" href="robocoin.pdf" target="_blank">{{ $t('header.navs.report') }}</a>
        </div>
        <div class="nav-link">
          <img class="nav-icon" src="/icons/download.svg" alt="Download Icon" />
          <a class="nav-item" href="https://flagopen.github.io/DataManage/" target="_blank">{{ $t('header.navs.download') }}</a>
        </div>
        <div class="nav-link">
          <img class="nav-icon" src="/icons/github.svg" alt="GitHub Icon" />
          <a class="nav-item" href="https://github.com/FlagOpen/RoboCOIN" target="_blank">{{ $t('header.navs.github') }}</a>
        </div>
        <!-- <div class="nav-link">
          <img class="nav-icon" src="/icons/visualize.svg" alt="Visualization Icon" />
          <a class="nav-item" href="#" target="_blank">{{ $t('header.navs.visualization') }}</a>
        </div> -->
        <div class="nav-link">
          <img class="nav-icon" src="/icons/about.svg" alt="About Us Icon" />
          <a class="nav-item" href="https://www.baai.ac.cn/zh-cn/" target="_blank">{{ $t('header.navs.about') }}</a>
        </div>
      </nav>

        <!-- 语言切换按钮 -->
      <div class="language-switcher">
        <button 
          class="lang-btn" 
          :class="{ active: currentLanguage === 'en' }"
          @click="switchLanguage('en')"
        >
          English
        </button>
        <button 
          class="lang-btn" 
          :class="{ active: currentLanguage === 'zh' }"
          @click="switchLanguage('zh')"
        >
          中文
        </button>
      </div> 
    </div>
  </header>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { locale, t } = useI18n()
const currentLanguage = ref('en') // 初始值已设为 en

// 切换语言函数
const switchLanguage = (lang) => {
  locale.value = lang
  currentLanguage.value = lang
  console.log('Switched language to:', lang)
  localStorage.setItem('language', lang)
  window.location.reload()
}

onMounted(() => {
  const savedLanguage = localStorage.getItem('language')
  if (savedLanguage) {
    // 有保存的语言，使用保存的
    locale.value = savedLanguage
    currentLanguage.value = savedLanguage
  } else {
    // 无保存的语言，强制使用 en 并保存到本地
    locale.value = 'en'
    currentLanguage.value = 'en'
    localStorage.setItem('language', 'en') // 首次加载时保存默认值
  }
})
</script>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: rgb(31, 115, 238);
  color: white;
  position: fixed;
  top: 0;
  width: 100%;
  height: 60px;
  z-index: 1000;
  padding: 0.5rem;
}

.logo {
  font-size: 1.8rem;
  font-weight: bold;
  display: flex;
  align-items: center;
  margin-left: 1rem;
}

.logo-image {
  height: 40px;
  margin-right: 1rem;
}

.logo-image2 {
  height: 50px;
  margin-left: 1rem;
}

.header-right {
  display: flex;
  align-items: center;
}

.nav-links {
  display: flex;
  gap: 1rem;
}

.nav-link {
  display: flex;
  align-items: center;
  padding: 0 1rem;
}

.nav-link:hover {
  background-color: rgba(255, 255, 255, 0.1);
}

.nav-item {
  color: white;
  text-decoration: none;
  font-size: 1.2rem;
  font-weight: 500;
  transition: color 0.3s;
}

.nav-icon {
  height: 25px;
  width: 25px;
  margin-right: 0.5rem;
}

.language-switcher {
  display: flex;
  gap: 0.5rem;
  margin-right: 1rem;
  margin-left: 2rem;
}

.lang-btn {
  padding: 6px 12px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.3);
  border-radius: 15px;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    padding: 1rem;
  }
  
  .nav-links {
    margin-top: 1rem;
    gap: 1rem;
  }
}
</style>