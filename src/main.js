// import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import ECharts from 'vue-echarts'
// vue-echarts
import { use } from "echarts/core"
import i18n from './locales'
import VuePdfEmbed from 'vue-pdf-embed'

import {
  CanvasRenderer
} from 'echarts/renderers'
import {
  BarChart,
  PieChart
} from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent
} from 'echarts/components'

use([
  CanvasRenderer,
  BarChart,
  PieChart,
  GridComponent,
  TooltipComponent
]);


const app = createApp(App)

app.use(router)
app.use(ElementPlus)
app.component('v-chart', ECharts)
app.component('vue-pdf-embed', VuePdfEmbed)
app.use(i18n)

app.mount('#app')