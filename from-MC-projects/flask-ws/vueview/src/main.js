// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import VueRouter from 'vue-router'
import VueSocketio from 'vue-socket.io'
import App from './App'
import store from './store'
// Components
import HelloWorld from './components/HelloWorld'
import Annotate1 from './components/Annotate1'
import Test2 from './components/Test2'
// could https://vuejs.org/v2/guide/components-registration.html#Automatic-Global-Registration-of-Base-Components

Vue.config.productionTip = false
Vue.use(VueRouter)
Vue.use(VueSocketio, `//${window.document.domain}:5000`, store)

let r = (path, component) => ({path, component})

const router = new VueRouter({
  routes: [
    r('/', HelloWorld),
    r('/annotate1', Annotate1),
    r('/test2', Test2)
  ]
})

/* eslint-disable no-new */
new Vue({
  el: '#app',
  components: { App },
  router,
  store,
  template: '<App/>'
})
