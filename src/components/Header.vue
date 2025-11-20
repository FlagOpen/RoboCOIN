<template>
  <header class="header">
    <div class="logo">
      <img class="logo-image" src="/logos/robocoin.png" alt="RoboCOIN Logo" />
    </div>
    
    <div class="header-right">
      <nav class="nav-links">
        <div class="nav-link">
          <img class="nav-icon" src="/icons/pdf.svg" alt="Technical Report Icon" />
          <a class="nav-item" href="robocoin.pdf" target="_blank">{{ $t('header.navs.report') }}</a>
        </div>
        <div class="nav-link">
          <img class="nav-icon" src="/icons/download.svg" alt="Download Icon" />
          <a class="nav-item" href="https://flagopen.github.io/RoboCOIN-DataManage/" target="_blank">{{ $t('header.navs.download') }}</a>
        </div>
        <div class="nav-link">
          <img class="nav-icon" src="/icons/github.svg" alt="GitHub Icon" />
          <a class="nav-item" href="https://github.com/FlagOpen/RoboCOIN" target="_blank">{{ $t('header.navs.github') }}</a>
        </div>
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
const currentLanguage = ref('en')

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
    locale.value = savedLanguage
    currentLanguage.value = savedLanguage
  } else {
    locale.value = 'en'
    currentLanguage.value = 'en'
    localStorage.setItem('language', 'en')
  }
})
</script>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  /* 浅蓝到深蓝渐变背景 */
  background: linear-gradient(135deg, #429dc1 0%, #1E40AF 100%);
  /* 毛玻璃效果 */
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  color: white;
  position: fixed;
  top: 0;
  width: 100%;
  height: 70px;
  z-index: 1000;
  padding: 0 2rem;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.logo {
  font-size: 1.8rem;
  font-weight: bold;
  display: flex;
  align-items: center;
  transition: transform 0.3s ease;
}

.logo:hover {
  transform: scale(1.05);
}

.logo-image {
  height: 45px;
  transition: all 0.3s ease;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.nav-links {
  display: flex;
  gap: 0.5rem;
}

.nav-link {
  display: flex;
  align-items: center;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  transition: all 0.3s ease;
  text-decoration: none;
  position: relative;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(5px);
}

.nav-link::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.nav-link:hover::before {
  left: 100%;
}

.nav-link:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
}

.nav-item {
  color: white;
  text-decoration: none;
  font-size: 1.1rem;
  font-weight: 500;
  transition: all 0.3s ease;
  position: relative;
}

.nav-link:hover .nav-item {
  color: #f0f8ff;
  text-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
}

.nav-icon {
  height: 20px;
  width: 20px;
  margin-right: 0.5rem;
  transition: all 0.3s ease;
  filter: brightness(0.9);
}

.nav-link:hover .nav-icon {
  transform: scale(1.1);
  filter: brightness(1.1);
}

.language-switcher {
  display: flex;
  gap: 0.5rem;
  background: rgba(255, 255, 255, 0.15);
  padding: 0.25rem;
  border-radius: 25px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.lang-btn {
  padding: 8px 20px;
  background: transparent;
  color: rgba(255, 255, 255, 0.9);
  border: none;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
}

.lang-btn.active, .lang-btn:hover {
  background: #fff;
  color: #1E40AF;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

/* 响应式设计 */
@media (max-width: 968px) {
  .header {
    padding: 0 1rem;
    height: 60px;
  }
  
  .nav-links {
    gap: 0.25rem;
  }
  
  .nav-link {
    padding: 0.4rem 0.8rem;
  }
  
  .nav-item {
    font-size: 1rem;
  }
}

@media (max-width: 768px) {
  .header {
    flex-direction: column;
    height: auto;
    padding: 1rem;
    position: relative;
    backdrop-filter: blur(8px) saturate(140%);
    -webkit-backdrop-filter: blur(8px) saturate(140%);
  }
  
  .header-right {
    width: 100%;
    justify-content: center;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  
  .nav-links {
    order: 2;
    width: 100%;
    justify-content: center;
    margin: 1rem 0;
  }
  
  .language-switcher {
    order: 1;
  }
  
  .logo {
    margin-bottom: 1rem;
  }
}

@media (max-width: 480px) {
  .nav-links {
    flex-direction: column;
    align-items: center;
  }
  
  .nav-link {
    width: 200px;
    justify-content: center;
  }
}

/* 浏览器兼容性回退 */
@supports not (backdrop-filter: blur(1px)) {
  .header {
    background: linear-gradient(135deg, #6ba8e0 0%, #1a3999 100%);
  }
  .nav-link {
    background: rgba(255, 255, 255, 0.2);
  }
}
</style>