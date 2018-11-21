import Vue from 'vue'
import Router from 'vue-router'
// import Home from './views/Home.vue'

// Components
import HelloWorld from './components/HelloWorld'
import Annotate1 from './components/Annotate1'
import Test2 from './components/Test2'
import ManualLabel from './components/ManualLabel'

Vue.use(Router)

let r = (path, component) => ({ path, component })

export default new Router({
  routes: [
    r('/', HelloWorld),
    r('/manual-label', ManualLabel),
    r('/annotate1', Annotate1),
    r('/test2', Test2),
    /*
    {
      path: '/',
      name: 'home',
      component: Home
    },
    */
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
