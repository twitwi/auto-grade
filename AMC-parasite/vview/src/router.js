import Vue from 'vue'
import Router from 'vue-router'
// import Home from './views/Home.vue'

// Components
import ManualLabel from './components/ManualLabel'
import MinisetBuilder from './components/MinisetBuilder'
import MassFix from './components/MassFix'
import ExamId from './components/ExamId'
import OCRQuestions from './components/OCRQuestions'

Vue.use(Router)

let r = (path, component) => ({ path, component })

export default new Router({
  routes: [
    r('/manual-label', ManualLabel),
    r('/miniset-builder', MinisetBuilder),
    r('/mass-fix', MassFix),
    r('/exam-id', ExamId),
    r('/ocr-questions', OCRQuestions),
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue')
    }
  ]
})
